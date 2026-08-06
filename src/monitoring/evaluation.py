"""Score stored predictions against realized prices.

The ``predictions`` table was designed for accuracy evaluation but nothing ever
filled ``actual_price``, so model accuracy was never measured. This module
closes the loop:

1. :func:`backfill_actual_prices` matches elapsed predictions with the realized
   ``market_history`` price for their ``target_time``.
2. :func:`compute_model_performance` aggregates the scored rows per
   (item, region, model version, horizon) into ``model_performance``.
"""

import logging
from datetime import timedelta
from typing import Dict, List, Optional

import numpy as np
from sqlalchemy import and_, exists, select, update

from src.config import settings
from src.database import MarketHistory, ModelPerformance, Prediction, get_session
from src.database.upsert import bulk_upsert
from src.models.train import regression_metrics
from src.utils.timeutils import utcnow

logger = logging.getLogger(__name__)


def _realized_price_subquery():
    """Correlated lookup of the realized average price for a prediction."""
    return (
        select(MarketHistory.average)
        .where(MarketHistory.time == Prediction.target_time)
        .where(MarketHistory.type_id == Prediction.type_id)
        .where(MarketHistory.region_id == Prediction.region_id)
        .limit(1)
        .scalar_subquery()
    )


async def backfill_actual_prices(max_age_days: int = 120) -> int:
    """Fill ``actual_price`` for predictions whose horizon has elapsed.

    Args:
        max_age_days: Ignore predictions older than this (they will never be
            matched if the history is already gone).

    Returns:
        Number of prediction rows scored.
    """
    now = utcnow()
    cutoff = now - timedelta(days=max_age_days)

    has_history = exists(
        select(MarketHistory.time).where(
            and_(
                MarketHistory.time == Prediction.target_time,
                MarketHistory.type_id == Prediction.type_id,
                MarketHistory.region_id == Prediction.region_id,
                MarketHistory.average.is_not(None),
            )
        )
    )

    statement = (
        update(Prediction)
        .where(Prediction.actual_price.is_(None))
        .where(Prediction.target_time <= now)
        .where(Prediction.target_time >= cutoff)
        .where(has_history)
        .values(actual_price=_realized_price_subquery())
    )

    async with get_session() as session:
        result = await session.execute(statement)
        await session.commit()

    scored = result.rowcount or 0
    logger.info(f"Backfilled actual_price for {scored} predictions")
    return scored


async def compute_model_performance(max_age_days: int = 120) -> List[Dict[str, float]]:
    """Aggregate scored predictions into the ``model_performance`` table."""
    cutoff = utcnow() - timedelta(days=max_age_days)

    async with get_session() as session:
        rows = (
            await session.execute(
                select(
                    Prediction.type_id,
                    Prediction.region_id,
                    Prediction.model_version,
                    Prediction.prediction_horizon,
                    Prediction.predicted_price,
                    Prediction.actual_price,
                )
                .where(Prediction.actual_price.is_not(None))
                .where(Prediction.prediction_time >= cutoff)
                .order_by(Prediction.prediction_time)
            )
        ).all()

    if not rows:
        logger.info("No scored predictions to aggregate yet")
        return []

    grouped: Dict[tuple, Dict[str, list]] = {}
    for type_id, region_id, version, horizon, predicted, actual in rows:
        key = (type_id, region_id, version or "unknown", horizon)
        bucket = grouped.setdefault(key, {"predicted": [], "actual": []})
        bucket["predicted"].append(float(predicted))
        bucket["actual"].append(float(actual))

    now = utcnow()
    records = []
    summaries = []
    for (type_id, region_id, version, horizon), values in grouped.items():
        metrics = regression_metrics(
            np.array(values["actual"]), np.array(values["predicted"])
        )
        record = {
            "time": now,
            "type_id": type_id,
            "region_id": region_id,
            "model_version": version[:50],
            "horizon_days": horizon,
            "n_samples": len(values["actual"]),
            "mae": _nan_to_none(metrics["mae"]),
            "rmse": _nan_to_none(metrics["rmse"]),
            "mape": _nan_to_none(metrics["mape"]),
            "directional_accuracy": _nan_to_none(metrics["directional_accuracy"]),
        }
        records.append(record)
        summaries.append(record)

    async with get_session() as session:
        await bulk_upsert(session, ModelPerformance, records)
        await session.commit()

    logger.info(f"Recorded performance for {len(records)} model/horizon combinations")
    return summaries


def _nan_to_none(value: float) -> Optional[float]:
    return None if value is None or np.isnan(value) else float(value)


async def recent_prediction_error(
    type_id: int,
    region_id: int,
    window_days: int = 7,
    horizon_days: int = 1,
) -> Optional[float]:
    """MAPE of scored predictions for one item over the last ``window_days``."""
    cutoff = utcnow() - timedelta(days=window_days)

    async with get_session() as session:
        rows = (
            await session.execute(
                select(Prediction.predicted_price, Prediction.actual_price)
                .where(Prediction.type_id == type_id)
                .where(Prediction.region_id == region_id)
                .where(Prediction.prediction_horizon == horizon_days)
                .where(Prediction.actual_price.is_not(None))
                .where(Prediction.prediction_time >= cutoff)
            )
        ).all()

    if not rows:
        return None

    predicted = np.array([float(p) for p, _ in rows])
    actual = np.array([float(a) for _, a in rows])
    mape = regression_metrics(actual, predicted)["mape"]
    return None if np.isnan(mape) else float(mape)


async def run_evaluation() -> Dict[str, object]:
    """Full evaluation pass, used by the scheduler and the CLI script."""
    scored = await backfill_actual_prices()
    performance = await compute_model_performance()
    return {
        "scored_predictions": scored,
        "performance_rows": len(performance),
        "performance": performance,
        "drift_threshold": settings.drift_threshold,
    }
