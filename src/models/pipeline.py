"""End-to-end training and prediction pipeline.

Single implementation shared by ``scripts/train_model.py``,
``scripts/predict_item.py``, ``scripts/predict_all_items.py``, the FastAPI
service and the scheduler's retraining job -- previously each of those had its
own copy with its own bugs.

The model is univariate by design: it consumes the last ``lookback_window``
daily average prices, MinMax-scaled with a scaler fit on the *training*
portion only, and predicts the next day. Multi-day forecasts are produced
autoregressively (each prediction is fed back as input), which compounds error
with horizon -- treat 30-day numbers as a trend indication, not a quote.
``ModelMetadata.feature_columns`` records the input columns so a multivariate
version can be introduced without breaking stored artifacts.
"""

import logging
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from sqlalchemy import select
from torch.utils.data import DataLoader

from src.config import settings
from src.constants import item_name
from src.database import MarketHistory, Prediction, get_session
from src.database.upsert import bulk_upsert
from src.models.artifacts import (
    ModelMetadata,
    ScalerParams,
    available_models,
    load_artifacts,
    load_metadata,
    make_model_version,
    save_artifacts,
)
from src.models.baselines import evaluate_baselines
from src.models.lstm_model import MarketDataset
from src.models.train import ModelTrainer, regression_metrics
from src.utils.timeutils import utcnow

logger = logging.getLogger(__name__)

#: Input columns the model is trained on. Univariate today; extending this list
#: is the entry point for a multivariate model.
FEATURE_COLUMNS: List[str] = ["average"]

MIN_TRAINING_ROWS = 50


class InsufficientData(RuntimeError):
    """Not enough history to train or forecast."""


class ModelNotTrained(RuntimeError):
    """No artifacts exist for the requested item."""


@dataclass
class TrainingResult:
    type_id: int
    region_id: int
    item_name: str
    promoted: bool
    model_version: str
    metrics: Dict[str, float]
    baselines: Dict[str, Dict[str, float]]
    previous_metrics: Optional[Dict[str, float]] = None
    n_train: int = 0
    n_val: int = 0


async def load_price_history(type_id: int, region_id: int) -> pd.DataFrame:
    """Daily average prices for an item, oldest first, zero prices dropped."""
    async with get_session() as session:
        records = (
            await session.execute(
                select(MarketHistory)
                .where(MarketHistory.type_id == type_id)
                .where(MarketHistory.region_id == region_id)
                .order_by(MarketHistory.time)
            )
        ).scalars().all()

    df = pd.DataFrame(
        [
            {
                "time": r.time,
                "average": float(r.average) if r.average else 0.0,
                "volume": r.volume or 0,
            }
            for r in records
        ]
    )

    if df.empty:
        return df
    return df[df["average"] > 0].reset_index(drop=True)


def build_sequences(scaled: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
    """Sliding windows: ``X[i] = scaled[i:i+lookback]``, ``y[i] = scaled[i+lookback]``."""
    sequences = np.array([scaled[i - lookback:i] for i in range(lookback, len(scaled))])
    targets = np.array([scaled[i][0] for i in range(lookback, len(scaled))])
    return sequences, targets


def _better(new: Dict[str, float], old: Optional[Dict[str, float]]) -> bool:
    """Promote only if validation error actually improved."""
    if not old:
        return True
    for key in ("val_mape", "val_rmse", "val_mae"):
        new_value, old_value = new.get(key), old.get(key)
        if new_value is None or old_value is None:
            continue
        if np.isnan(new_value) or np.isnan(old_value):
            continue
        return new_value < old_value
    return True


async def train_item(
    type_id: int,
    region_id: Optional[int] = None,
    epochs: int = 50,
    lookback: Optional[int] = None,
    val_split: float = 0.2,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.2,
    batch_size: int = 32,
    promote_only_if_better: bool = False,
) -> TrainingResult:
    """Train (and conditionally promote) the model for one item.

    Args:
        promote_only_if_better: Used by the weekly retraining job -- keep the
            existing checkpoint unless the new one validates better.
    """
    region_id = region_id or settings.region_ids[0]
    lookback = lookback or settings.lookback_window
    name = item_name(type_id)

    df = await load_price_history(type_id, region_id)
    if len(df) < max(MIN_TRAINING_ROWS, lookback + 10):
        raise InsufficientData(
            f"{name} ({type_id}) has {len(df)} usable rows; "
            f"need at least {max(MIN_TRAINING_ROWS, lookback + 10)}"
        )

    prices = df[FEATURE_COLUMNS].to_numpy(dtype=float)

    total_samples = len(prices) - lookback
    val_size = int(total_samples * val_split)
    train_samples = total_samples - val_size
    if train_samples <= 0:
        raise InsufficientData(f"{name} ({type_id}): validation split leaves no training data")

    # Fit the scaler on the training portion only. Fitting on the whole series
    # leaks the validation period's min/max into training.
    train_rows = train_samples + lookback
    scaler = MinMaxScaler()
    scaler.fit(prices[:train_rows])
    scaled = scaler.transform(prices)

    sequences, targets = build_sequences(scaled, lookback)
    dataset = MarketDataset(sequences, targets)

    trainer = ModelTrainer(
        input_size=len(FEATURE_COLUMNS),
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        epochs=epochs,
        batch_size=batch_size,
    )

    registered_name = (
        f"{settings.mlflow_registered_model_prefix}-{type_id}-{region_id}"
    )
    history = trainer.train(
        dataset,
        val_split=val_split,
        run_name=f"{name}-{type_id}",
        extra_params={"type_id": type_id, "region_id": region_id, "item_name": name},
        registered_model_name=registered_name,
    )

    val_loader = DataLoader(history["val_dataset"], batch_size=batch_size, shuffle=False)

    # Scaled-space metrics, comparable across items.
    scaled_metrics = trainer.evaluate(val_loader)

    # The same validation window in ISK, which is what a human reads.
    scaled_targets, scaled_predictions = trainer.predict_loader(val_loader)
    if scaled_targets.size:
        actual = scaler.inverse_transform(scaled_targets.reshape(-1, 1)).ravel()
        predicted = scaler.inverse_transform(scaled_predictions.reshape(-1, 1)).ravel()
        val_metrics = regression_metrics(actual, predicted)
    else:
        val_metrics = regression_metrics(np.array([]), np.array([]))

    metrics = {
        "best_val_loss": float(history["best_val_loss"]),
        "val_mae": val_metrics["mae"],
        "val_rmse": val_metrics["rmse"],
        "val_mape": val_metrics["mape"],
        "val_directional_accuracy": val_metrics["directional_accuracy"],
        "val_mae_scaled": scaled_metrics["mae"],
        "val_rmse_scaled": scaled_metrics["rmse"],
    }

    baselines = evaluate_baselines(prices[:, 0], lookback, train_samples)
    for baseline_name, baseline_metrics in baselines.items():
        logger.info(
            f"Baseline {baseline_name}: MAPE {baseline_metrics['mape']:.2f}% "
            f"(model {metrics['val_mape']:.2f}%)"
        )

    previous = load_metadata(type_id, region_id)
    previous_metrics = previous.metrics if previous else None

    promoted = True
    if promote_only_if_better and not _better(metrics, previous_metrics):
        promoted = False
        logger.warning(
            f"{name}: new model (val MAPE {metrics['val_mape']:.2f}%) did not beat the "
            f"incumbent ({(previous_metrics or {}).get('val_mape')}); keeping the old one"
        )

    trained_at = utcnow()
    version = (
        make_model_version(type_id, region_id, trained_at)
        if promoted
        else (previous.model_version if previous else "")
    )

    if promoted:
        metadata = ModelMetadata(
            type_id=type_id,
            region_id=region_id,
            model_version=version,
            trained_at=trained_at.isoformat(),
            lookback_window=lookback,
            horizon_days=settings.prediction_horizon_days,
            feature_columns=list(FEATURE_COLUMNS),
            input_size=len(FEATURE_COLUMNS),
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            scaler=ScalerParams.from_scaler(scaler),
            metrics=metrics,
            baselines=baselines,
            n_train_samples=int(history["n_train"]),
            n_val_samples=int(history["n_val"]),
            train_end=str(df["time"].iloc[train_rows - 1]),
        )
        save_artifacts(trainer.model.state_dict(), metadata)

    return TrainingResult(
        type_id=type_id,
        region_id=region_id,
        item_name=name,
        promoted=promoted,
        model_version=version,
        metrics=metrics,
        baselines=baselines,
        previous_metrics=previous_metrics,
        n_train=int(history["n_train"]),
        n_val=int(history["n_val"]),
    )


def _confidence_from_metrics(metrics: Dict[str, float]) -> Optional[float]:
    """Confidence == 1 - validation MAPE, clipped to [0, 1].

    Not a calibrated probability, but it is derived from measured error rather
    than the hardcoded 0.85 the API used to report.
    """
    mape = metrics.get("val_mape")
    if mape is None or np.isnan(mape):
        return None
    return round(float(np.clip(1.0 - mape / 100.0, 0.0, 1.0)), 4)


def forecast_from_prices(
    model,
    metadata: ModelMetadata,
    prices: np.ndarray,
    horizon_days: int,
) -> np.ndarray:
    """Autoregressive multi-day forecast in original units.

    Args:
        prices: At least ``lookback_window`` recent prices, oldest first.
    """
    lookback = metadata.lookback_window
    if len(prices) < lookback:
        raise InsufficientData(
            f"Need {lookback} prices to forecast, got {len(prices)}"
        )

    scaler = metadata.build_scaler()
    scaled = scaler.transform(np.asarray(prices, dtype=float).reshape(-1, 1))
    window = torch.FloatTensor(scaled[-lookback:].reshape(1, lookback, metadata.input_size))

    predictions = []
    model.eval()
    with torch.no_grad():
        for _ in range(horizon_days):
            value = float(model(window).cpu().numpy()[0][0])
            predictions.append(value)
            window = torch.cat(
                [window[:, 1:, :], torch.FloatTensor([[[value]]])], dim=1
            )

    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).ravel()


async def generate_predictions(
    type_id: int,
    region_id: Optional[int] = None,
    horizon_days: Optional[int] = None,
    store: bool = True,
) -> Dict[str, Any]:
    """Forecast from the *saved* model and persist the result.

    Every prediction path goes through here, so the ``predictions`` hypertable
    is always populated -- it is the basis for accuracy evaluation, the
    dashboard overlay and drift-versus-accuracy analysis.
    """
    region_id = region_id or settings.region_ids[0]
    horizon_days = horizon_days or settings.prediction_horizon_days

    loaded = load_artifacts(type_id, region_id)
    if loaded is None:
        raise ModelNotTrained(
            f"No trained model for type_id={type_id} region_id={region_id}. "
            f"Run scripts/train_model.py --type-id {type_id} first."
        )
    model, metadata = loaded

    df = await load_price_history(type_id, region_id)
    if len(df) < metadata.lookback_window:
        raise InsufficientData(
            f"Need {metadata.lookback_window} price points, have {len(df)}"
        )

    prices = df["average"].to_numpy(dtype=float)
    predicted = forecast_from_prices(model, metadata, prices, horizon_days)

    last_date = pd.Timestamp(df["time"].iloc[-1]).to_pydatetime()
    current_price = float(prices[-1])
    prediction_time = utcnow()
    confidence = _confidence_from_metrics(metadata.metrics)

    horizons = [
        {
            "horizon_days": day + 1,
            "target_time": last_date + timedelta(days=day + 1),
            "predicted_price": float(predicted[day]),
        }
        for day in range(horizon_days)
    ]

    if store:
        rows = [
            {
                "prediction_time": prediction_time,
                "type_id": type_id,
                "region_id": region_id,
                "prediction_horizon": h["horizon_days"],
                "target_time": h["target_time"],
                "current_price": round(current_price, 2),
                "predicted_price": round(h["predicted_price"], 2),
                "model_version": metadata.model_version,
                "confidence": confidence,
            }
            for h in horizons
        ]
        async with get_session() as session:
            await bulk_upsert(session, Prediction, rows)
            await session.commit()
        logger.info(f"Stored {len(rows)} prediction rows for type_id={type_id}")

    final = horizons[-1]["predicted_price"]
    return {
        "item_name": item_name(type_id),
        "type_id": type_id,
        "region_id": region_id,
        "model_version": metadata.model_version,
        "prediction_time": prediction_time,
        "last_observed_date": last_date,
        "current_price": current_price,
        "predicted_price": final,
        "percent_change": (final / current_price - 1) * 100 if current_price else float("nan"),
        "confidence": confidence,
        "horizon_days": horizon_days,
        "predictions": [
            {
                "day": h["horizon_days"],
                "date": h["target_time"].strftime("%Y-%m-%d"),
                "predicted_price": h["predicted_price"],
            }
            for h in horizons
        ],
        "model_metrics": metadata.metrics,
    }


def trained_items() -> List[Tuple[int, int]]:
    """``(type_id, region_id)`` pairs that currently have artifacts."""
    return available_models()
