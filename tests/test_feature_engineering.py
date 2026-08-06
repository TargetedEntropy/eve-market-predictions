"""Feature pipeline (issue 1.2: float(Timestamp) crashed every scheduled run)."""

from datetime import datetime

import numpy as np
import pandas as pd

from src.features.feature_engineering import FeatureEngine
from src.features.technical_indicators import TechnicalIndicators


def test_numeric_features_drops_the_time_column(history_frame):
    """The regression that broke the hourly job: iloc[-1] includes a Timestamp."""
    computed = TechnicalIndicators.compute_all(history_frame.copy())
    row = computed.iloc[-1].to_dict()
    assert isinstance(row["time"], pd.Timestamp)

    features = FeatureEngine._numeric_features(row)

    assert "time" not in features
    assert all(isinstance(v, float) for v in features.values())
    assert "average" in features and "rsi_14" in features


def test_numeric_features_drops_nan_and_inf():
    row = {"good": 1.5, "nan": float("nan"), "inf": float("inf"),
           "neg_inf": float("-inf"), "none": None}
    assert FeatureEngine._numeric_features(row) == {"good": 1.5}


def test_numeric_features_drops_strings_and_datetimes():
    row = {"name": "PLEX", "when": datetime(2025, 1, 1), "value": 3}
    assert FeatureEngine._numeric_features(row) == {"value": 3.0}


def test_numeric_features_keeps_booleans_as_floats():
    assert FeatureEngine._numeric_features({"flag": True, "np_flag": np.bool_(False)}) == {
        "flag": 1.0,
        "np_flag": 0.0,
    }


def test_numeric_features_are_json_and_db_safe(history_frame):
    """Every value must survive float() -- that is what goes into Numeric."""
    computed = TechnicalIndicators.compute_all(history_frame.copy())
    for value in FeatureEngine._numeric_features(computed.iloc[-1].to_dict()).values():
        assert float(value) == value


def test_order_book_features():
    orders = pd.DataFrame(
        [
            {"price": 100.0, "volume_remain": 10, "is_buy_order": True},
            {"price": 99.0, "volume_remain": 5, "is_buy_order": True},
            {"price": 110.0, "volume_remain": 20, "is_buy_order": False},
            {"price": 115.0, "volume_remain": 7, "is_buy_order": False},
        ]
    )
    features = FeatureEngine.compute_order_book_features(orders)

    assert features["best_bid"] == 100.0
    assert features["best_ask"] == 110.0
    assert features["spread_abs"] == 10.0
    assert features["mid_price"] == 105.0
    assert features["bid_volume"] == 15.0
    assert features["ask_volume"] == 27.0
    assert features["total_orders"] == 4.0
    assert features["imbalance"] == (15 - 27) / 42


def test_order_book_features_on_one_sided_book():
    orders = pd.DataFrame([{"price": 100.0, "volume_remain": 10, "is_buy_order": True}])
    features = FeatureEngine.compute_order_book_features(orders)
    assert features["best_ask"] == 0.0
    assert features["spread_pct"] == 0.0
    assert features["imbalance"] == 1.0


def test_order_book_features_on_empty_book():
    assert FeatureEngine.compute_order_book_features(pd.DataFrame()) == {}


def test_temporal_features_are_cyclic_and_bounded():
    features = FeatureEngine.compute_temporal_features(datetime(2025, 6, 7, 20, 0))
    assert features["is_weekend"] == 1.0  # Saturday
    assert features["is_peak_hour"] == 1.0  # 20:00
    for name in ("hour_sin", "hour_cos", "dow_sin", "dow_cos", "dom_sin", "dom_cos"):
        assert -1.0 <= features[name] <= 1.0
    assert all(isinstance(v, float) for v in features.values())


def test_temporal_features_weekday_off_peak():
    features = FeatureEngine.compute_temporal_features(datetime(2025, 6, 4, 9, 0))
    assert features["is_weekend"] == 0.0
    assert features["is_peak_hour"] == 0.0
