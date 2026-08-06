"""Technical indicators (issue 9.2: two library branches produced different features)."""

import pandas as pd

from src.features.technical_indicators import INDICATOR_COLUMNS, TechnicalIndicators


def test_compute_all_adds_exactly_the_documented_columns(history_frame):
    before = set(history_frame.columns)
    out = TechnicalIndicators.compute_all(history_frame.copy())
    added = set(out.columns) - before
    assert added == set(INDICATOR_COLUMNS)


def test_indicator_columns_has_no_duplicates():
    assert len(INDICATOR_COLUMNS) == len(set(INDICATOR_COLUMNS))


def test_ema_7_is_present(history_frame):
    """The live 'ta' branch used to silently omit ema_7."""
    out = TechnicalIndicators.compute_all(history_frame.copy())
    assert "ema_7" in out.columns
    assert out["ema_7"].notna().any()


def test_latest_row_is_fully_populated(history_frame):
    """With 300 days of history nothing should still be warming up."""
    out = TechnicalIndicators.compute_all(history_frame.copy())
    assert out.iloc[-1].isna().sum() == 0


def test_works_with_close_column_instead_of_average():
    df = pd.DataFrame({"close": [float(v) for v in range(100, 200)]})
    out = TechnicalIndicators.compute_all(df)
    assert "sma_7" in out.columns
    assert out["sma_7"].notna().any()


def test_missing_volume_column_is_tolerated():
    df = pd.DataFrame({"average": [float(v) for v in range(100, 200)]})
    out = TechnicalIndicators.compute_all(df)
    assert "volume_sma_7" not in out.columns
    assert "rsi_14" in out.columns
