"""Shared pytest fixtures.

Tests never touch PostgreSQL, Redis, MLflow or ESI: the DB-backed paths are
exercised through fakes, and everything else runs on synthetic data. Anything
that genuinely needs a live database is marked ``@pytest.mark.integration``.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# Point config at throwaway values before src.config is imported anywhere.
os.environ.setdefault("ESI_CACHE_ENABLED", "False")
os.environ.setdefault("MLFLOW_TRACKING_URI", "file:///tmp/eve-discovery-tests-mlruns")

from src.config import settings  # noqa: E402
from src.models import artifacts  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_model_dir(tmp_path, monkeypatch):
    """Give every test its own empty MODEL_DIR."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    monkeypatch.setattr(settings, "model_dir", str(model_dir))
    artifacts.clear_artifact_cache()
    yield model_dir
    artifacts.clear_artifact_cache()


@pytest.fixture
def no_sleep(monkeypatch):
    """Make asyncio.sleep instantaneous so backoff paths are testable."""
    import asyncio

    slept = []

    async def fake_sleep(seconds, *args, **kwargs):
        slept.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    return slept


@pytest.fixture
def make_metadata():
    """Factory for ModelMetadata objects with sane defaults."""
    from src.models.artifacts import ModelMetadata, ScalerParams
    from src.models.pipeline import FEATURE_COLUMNS

    def _make(type_id=44992, region_id=10000002, version="v1", metrics=None, scaler=None):
        return ModelMetadata(
            type_id=type_id,
            region_id=region_id,
            model_version=version,
            trained_at="2026-01-01T00:00:00",
            lookback_window=30,
            horizon_days=30,
            feature_columns=list(FEATURE_COLUMNS),
            input_size=1,
            hidden_size=16,
            num_layers=1,
            dropout=0.0,
            scaler=scaler or ScalerParams(data_min=[1_000_000.0], data_max=[9_000_000.0]),
            metrics=metrics or {"val_mape": 4.0},
        )

    return _make


@pytest.fixture
def tiny_model():
    """A small untrained MarketLSTM matching ``make_metadata``'s architecture."""
    from src.models.lstm_model import MarketLSTM

    def _make():
        return MarketLSTM(input_size=1, hidden_size=16, num_layers=1, dropout=0.0)

    return _make


@pytest.fixture
def price_series():
    """A deterministic 300-day price series with trend, cycle and noise."""
    import numpy as np

    rng = np.random.RandomState(42)
    n = 300
    trend = np.linspace(5_000_000, 6_500_000, n)
    cycle = 200_000 * np.sin(np.arange(n) / 12)
    noise = rng.normal(0, 30_000, n)
    return trend + cycle + noise


@pytest.fixture
def history_frame(price_series):
    """A market-history style DataFrame."""
    import numpy as np
    import pandas as pd

    rng = np.random.RandomState(7)
    df = pd.DataFrame(
        {
            "time": pd.date_range("2025-01-01", periods=len(price_series), freq="D"),
            "average": price_series,
        }
    )
    df["highest"] = df["average"] * 1.03
    df["lowest"] = df["average"] * 0.97
    df["volume"] = rng.randint(10_000, 90_000, len(df))
    df["order_count"] = rng.randint(50, 500, len(df))
    return df
