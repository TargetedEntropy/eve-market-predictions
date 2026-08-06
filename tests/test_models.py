"""Training, artifacts and forecasting (issues 1.4, 1.5, 1.8, 1.9, 7.3)."""

import json

import numpy as np
import pytest
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

from src.models.artifacts import (
    ScalerParams,
    available_models,
    checkpoint_path,
    clear_artifact_cache,
    load_artifacts,
    load_metadata,
    make_model_version,
    metadata_path,
    save_artifacts,
)
from src.models.baselines import evaluate_baselines, linear_trend_forecast, naive_forecast
from src.models.lstm_model import MarketDataset, MarketLSTM
from src.models.pipeline import (
    InsufficientData,
    _better,
    _confidence_from_metrics,
    build_sequences,
    forecast_from_prices,
)
from src.models.train import ModelTrainer, chronological_split, regression_metrics

# --------------------------------------------------------------------------- #
# Metrics (1.9)
# --------------------------------------------------------------------------- #

def test_mape_ignores_zero_targets_instead_of_returning_inf():
    targets = np.array([0.0, 100.0, 200.0])
    predictions = np.array([5.0, 110.0, 180.0])
    metrics = regression_metrics(targets, predictions)

    assert np.isfinite(metrics["mape"])
    assert metrics["mape"] == pytest.approx((0.1 + 0.1) / 2 * 100)


def test_mape_is_nan_when_every_target_is_zero():
    metrics = regression_metrics(np.zeros(3), np.ones(3))
    assert np.isnan(metrics["mape"])
    assert np.isfinite(metrics["mae"])


def test_metrics_on_empty_input_do_not_raise():
    metrics = regression_metrics(np.array([]), np.array([]))
    assert all(np.isnan(v) for v in metrics.values())


def test_directional_accuracy():
    targets = np.array([1.0, 2.0, 3.0, 2.0])
    metrics = regression_metrics(targets, np.array([1.0, 1.5, 2.5, 1.0]))
    assert metrics["directional_accuracy"] == 100.0


# --------------------------------------------------------------------------- #
# Splitting (1.8)
# --------------------------------------------------------------------------- #

def test_chronological_split_is_ordered_and_disjoint():
    sequences = np.arange(100 * 3, dtype=float).reshape(100, 3, 1)
    dataset = MarketDataset(sequences, np.arange(100, dtype=float))

    train, val = chronological_split(dataset, 0.2)

    assert len(train) == 80 and len(val) == 20
    assert float(train[len(train) - 1][1]) == 79.0
    assert float(val[0][1]) == 80.0, "validation must be strictly after training"


def test_chronological_split_rejects_impossible_split():
    dataset = MarketDataset(np.zeros((2, 3, 1)), np.zeros(2))
    with pytest.raises(ValueError):
        chronological_split(dataset, 1.0)


def test_build_sequences_shapes():
    scaled = np.arange(50, dtype=float).reshape(-1, 1)
    sequences, targets = build_sequences(scaled, lookback=10)

    assert sequences.shape == (40, 10, 1)
    assert targets.shape == (40,)
    assert targets[0] == 10.0
    assert sequences[0][0] == 0.0


# --------------------------------------------------------------------------- #
# Scaler persistence (1.4)
# --------------------------------------------------------------------------- #

def test_scaler_params_round_trip_exactly(price_series):
    values = price_series.reshape(-1, 1)
    original = MinMaxScaler().fit(values[:200])

    restored = ScalerParams.from_scaler(original).to_scaler()

    np.testing.assert_allclose(original.transform(values), restored.transform(values))
    np.testing.assert_allclose(
        original.inverse_transform(original.transform(values)),
        restored.inverse_transform(restored.transform(values)),
    )


def test_scaler_params_survive_json():
    scaler = MinMaxScaler().fit(np.array([[1.0], [10.0]]))
    params = ScalerParams.from_scaler(scaler)
    revived = ScalerParams(**json.loads(json.dumps(
        {**params.__dict__, "feature_range": list(params.feature_range)}
    )))
    np.testing.assert_allclose(
        revived.to_scaler().transform([[5.0]]), scaler.transform([[5.0]])
    )


def test_scaler_handles_a_constant_series():
    """A flat price series has zero range; scaling must not divide by zero."""
    scaler = ScalerParams(data_min=[7.0], data_max=[7.0]).to_scaler()
    assert np.isfinite(scaler.transform([[7.0]])).all()


# --------------------------------------------------------------------------- #
# Artifacts (1.5, 7.3)
# --------------------------------------------------------------------------- #

def test_artifacts_are_per_item(isolated_model_dir, make_metadata):
    a = MarketLSTM(input_size=1, hidden_size=16, num_layers=1, dropout=0.0)
    b = MarketLSTM(input_size=1, hidden_size=16, num_layers=1, dropout=0.0)

    save_artifacts(a.state_dict(), make_metadata(44992, version="plex-1"))
    save_artifacts(b.state_dict(), make_metadata(34, version="trit-1"))

    assert load_metadata(44992, 10000002).model_version == "plex-1"
    assert load_metadata(34, 10000002).model_version == "trit-1", \
        "training one item must not clobber another's checkpoint"
    assert sorted(available_models()) == [(34, 10000002), (44992, 10000002)]


def test_checkpoint_loads_with_weights_only(isolated_model_dir, make_metadata):
    model = MarketLSTM(input_size=1, hidden_size=16, num_layers=1, dropout=0.0)
    save_artifacts(model.state_dict(), make_metadata())

    # weights_only=True is what load_artifacts uses; assert it works directly.
    state = torch.load(checkpoint_path(44992, 10000002), weights_only=True)
    assert isinstance(state, dict)
    assert all(isinstance(v, torch.Tensor) for v in state.values())


def test_metadata_is_plain_json(isolated_model_dir, make_metadata):
    model = MarketLSTM(input_size=1, hidden_size=16, num_layers=1, dropout=0.0)
    save_artifacts(model.state_dict(), make_metadata())

    payload = json.loads(metadata_path(44992, 10000002).read_text())
    assert payload["scaler"]["data_min"] == [1_000_000.0]
    assert payload["feature_columns"] == ["average"]


def test_load_artifacts_returns_none_for_unknown_item(isolated_model_dir):
    assert load_artifacts(1234, 10000002) is None


def test_load_artifacts_returns_none_when_checkpoint_missing(isolated_model_dir, make_metadata):
    model = MarketLSTM(input_size=1, hidden_size=16, num_layers=1, dropout=0.0)
    save_artifacts(model.state_dict(), make_metadata())
    checkpoint_path(44992, 10000002).unlink()
    clear_artifact_cache()
    assert load_artifacts(44992, 10000002) is None


def test_artifact_cache_refreshes_when_the_checkpoint_changes(isolated_model_dir, make_metadata):
    first = MarketLSTM(input_size=1, hidden_size=16, num_layers=1, dropout=0.0)
    save_artifacts(first.state_dict(), make_metadata(version="v1"))
    assert load_artifacts(44992, 10000002)[1].model_version == "v1"

    second = MarketLSTM(input_size=1, hidden_size=16, num_layers=1, dropout=0.0)
    save_artifacts(second.state_dict(), make_metadata(version="v2"))
    assert load_artifacts(44992, 10000002)[1].model_version == "v2"


def test_model_version_is_short_enough_for_the_column():
    version = make_model_version(44992, 10000002)
    assert len(version) <= 50
    assert version.startswith("44992-10000002-")


# --------------------------------------------------------------------------- #
# Forecasting (1.4)
# --------------------------------------------------------------------------- #

def test_forecast_returns_isk_not_scaled_values(isolated_model_dir, price_series, make_metadata):
    scaler = MinMaxScaler().fit(price_series.reshape(-1, 1))
    metadata = make_metadata()
    metadata.scaler = ScalerParams.from_scaler(scaler)

    model = MarketLSTM(input_size=1, hidden_size=16, num_layers=1, dropout=0.0)
    forecast = forecast_from_prices(model, metadata, price_series, horizon_days=5)

    assert forecast.shape == (5,)
    # An untrained net still outputs 0-1 space; the inverse transform must lift
    # results back into the price range rather than returning ~0.5.
    assert forecast.min() > price_series.min() * 0.1
    assert forecast.max() < price_series.max() * 3


def test_forecast_requires_a_full_lookback_window(isolated_model_dir, make_metadata):
    model = MarketLSTM(input_size=1, hidden_size=16, num_layers=1, dropout=0.0)
    with pytest.raises(InsufficientData):
        forecast_from_prices(model, make_metadata(), np.arange(5, dtype=float), 3)


def test_confidence_is_derived_from_validation_error():
    assert _confidence_from_metrics({"val_mape": 5.0}) == 0.95
    assert _confidence_from_metrics({"val_mape": 250.0}) == 0.0
    assert _confidence_from_metrics({}) is None
    assert _confidence_from_metrics({"val_mape": float("nan")}) is None


def test_promotion_requires_a_better_model():
    assert _better({"val_mape": 3.0}, {"val_mape": 5.0}) is True
    assert _better({"val_mape": 7.0}, {"val_mape": 5.0}) is False
    assert _better({"val_mape": 7.0}, None) is True
    assert _better({"val_mape": 7.0}, {}) is True


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #

def test_naive_and_linear_baselines():
    window = np.array([1.0, 2.0, 3.0, 4.0])
    assert naive_forecast(window) == 4.0
    assert linear_trend_forecast(window) == pytest.approx(5.0)


def test_evaluate_baselines_returns_metrics_in_price_units(price_series):
    results = evaluate_baselines(price_series, lookback=30, val_start_sample=200)

    assert set(results) == {"naive", "moving_average_7", "linear_trend"}
    for metrics in results.values():
        assert np.isfinite(metrics["mape"])
        assert metrics["mae"] > 0


def test_evaluate_baselines_handles_no_validation_samples():
    assert evaluate_baselines(np.arange(40, dtype=float), lookback=30, val_start_sample=99) == {}


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #

def test_training_restores_the_best_epoch_and_does_not_write_files(
    isolated_model_dir, price_series
):
    scaler = MinMaxScaler().fit(price_series[:200].reshape(-1, 1))
    sequences, targets = build_sequences(scaler.transform(price_series.reshape(-1, 1)), 30)

    trainer = ModelTrainer(input_size=1, hidden_size=16, num_layers=1, epochs=2, batch_size=16)
    history = trainer.train(MarketDataset(sequences, targets), val_split=0.2)

    assert trainer.best_state is not None
    assert history["n_train"] + history["n_val"] == len(sequences)
    # The trainer must not persist anything itself (that is artifacts.py's job).
    assert list(isolated_model_dir.iterdir()) == []


def test_training_survives_a_final_batch_of_one(isolated_model_dir):
    """outputs.squeeze() collapsed a 1-row batch to a scalar and broadcast."""
    sequences = np.random.RandomState(0).rand(33, 10, 1)
    targets = np.random.RandomState(1).rand(33)

    trainer = ModelTrainer(input_size=1, hidden_size=8, num_layers=1, epochs=1, batch_size=8)
    history = trainer.train(MarketDataset(sequences, targets), val_split=0.2)

    assert np.isfinite(history["best_val_loss"])


def test_evaluate_returns_all_four_metrics(isolated_model_dir):
    sequences = np.random.RandomState(0).rand(60, 10, 1)
    targets = np.random.RandomState(1).rand(60)
    dataset = MarketDataset(sequences, targets)

    trainer = ModelTrainer(input_size=1, hidden_size=8, num_layers=1, epochs=1, batch_size=16)
    trainer.train(dataset, val_split=0.2)

    metrics = trainer.evaluate(DataLoader(dataset, batch_size=16))
    assert set(metrics) == {"mae", "rmse", "mape", "directional_accuracy"}
    assert np.isfinite(metrics["mae"])


def test_evaluate_before_training_raises():
    with pytest.raises(RuntimeError):
        ModelTrainer(input_size=1).evaluate(
            DataLoader(MarketDataset(np.zeros((2, 3, 1)), np.zeros(2)))
        )
