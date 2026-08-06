"""Technical indicators.

One library, one code path: the ``ta`` package (pinned in requirements.txt).
The previous module tried ``pandas_ta`` first and silently produced a
*different* feature set -- including a missing ``ema_7`` -- depending on what
happened to be installed. Feature names are plain snake_case so they are stable
across library versions.

``compute_all`` adds :data:`INDICATOR_COLUMNS` columns to the frame.
"""

from typing import List

import numpy as np
import pandas as pd
import ta


def _price_column(df: pd.DataFrame) -> str:
    """Market history calls it ``average``; OHLC frames call it ``close``."""
    return "average" if "average" in df.columns else "close"


class TechnicalIndicators:
    """Compute technical indicators for price series"""

    @staticmethod
    def compute_price_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Moving averages, RSI, MACD and Bollinger Bands."""
        prices = df[_price_column(df)]

        df["sma_7"] = ta.trend.sma_indicator(prices, window=7)
        df["sma_14"] = ta.trend.sma_indicator(prices, window=14)
        df["sma_30"] = ta.trend.sma_indicator(prices, window=30)
        df["ema_7"] = ta.trend.ema_indicator(prices, window=7)
        df["ema_12"] = ta.trend.ema_indicator(prices, window=12)
        df["ema_26"] = ta.trend.ema_indicator(prices, window=26)
        df["rsi_14"] = ta.momentum.rsi(prices, window=14)

        macd = ta.trend.MACD(prices, window_slow=26, window_fast=12, window_sign=9)
        df["macd"] = macd.macd()
        df["macd_hist"] = macd.macd_diff()
        df["macd_signal"] = macd.macd_signal()

        bollinger = ta.volatility.BollingerBands(prices, window=20, window_dev=2)
        df["bb_lower"] = bollinger.bollinger_lband()
        df["bb_middle"] = bollinger.bollinger_mavg()
        df["bb_upper"] = bollinger.bollinger_hband()
        band_width = df["bb_upper"] - df["bb_lower"]
        df["bb_width"] = band_width / df["bb_middle"].replace(0, np.nan)
        df["bb_position"] = (prices - df["bb_lower"]) / band_width.replace(0, np.nan)

        return df

    @staticmethod
    def compute_range_features(df: pd.DataFrame) -> pd.DataFrame:
        """Intraday range features from the daily high/low/average."""
        if not {"highest", "lowest"}.issubset(df.columns):
            return df

        prices = df[_price_column(df)]
        spread = df["highest"] - df["lowest"]

        df["hl_spread"] = spread
        df["hl_spread_pct"] = spread / prices.replace(0, np.nan) * 100
        df["close_to_high"] = (df["highest"] - prices) / prices.replace(0, np.nan) * 100
        df["close_to_low"] = (prices - df["lowest"]) / prices.replace(0, np.nan) * 100

        return df

    @staticmethod
    def compute_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Volume moving averages, rate of change and on-balance volume."""
        if "volume" not in df.columns:
            return df

        volumes = df["volume"].astype(float)

        df["volume_sma_7"] = volumes.rolling(window=7).mean()
        df["volume_sma_14"] = volumes.rolling(window=14).mean()
        df["volume_roc"] = volumes.pct_change(periods=1)
        df["volume_ratio_7"] = volumes / df["volume_sma_7"].replace(0, np.nan)
        df["obv"] = ta.volume.on_balance_volume(df[_price_column(df)], volumes)

        return df

    @staticmethod
    def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
        """Simple/log returns and rolling volatility."""
        prices = df[_price_column(df)]

        df["return_1d"] = prices.pct_change(1)
        df["return_7d"] = prices.pct_change(7)
        df["return_14d"] = prices.pct_change(14)
        df["log_return_1d"] = np.log(prices / prices.shift(1))

        df["volatility_7d"] = df["return_1d"].rolling(window=7).std()
        df["volatility_14d"] = df["return_1d"].rolling(window=14).std()
        df["volatility_30d"] = df["return_1d"].rolling(window=30).std()

        return df

    @staticmethod
    def compute_momentum(df: pd.DataFrame) -> pd.DataFrame:
        """Rate of change and absolute momentum."""
        prices = df[_price_column(df)]

        df["roc_7"] = prices.pct_change(periods=7) * 100
        df["roc_14"] = prices.pct_change(periods=14) * 100
        df["momentum_7"] = prices.diff(periods=7)
        df["momentum_14"] = prices.diff(periods=14)

        return df

    @staticmethod
    def compute_all(df: pd.DataFrame) -> pd.DataFrame:
        """Compute every indicator. Returns the same frame, extended."""
        df = TechnicalIndicators.compute_price_indicators(df)
        df = TechnicalIndicators.compute_range_features(df)
        df = TechnicalIndicators.compute_volume_indicators(df)
        df = TechnicalIndicators.compute_returns(df)
        df = TechnicalIndicators.compute_momentum(df)

        return df


#: Every column ``compute_all`` adds, in order. Keep in sync with the methods
#: above -- ``tests/test_technical_indicators.py`` asserts they match.
INDICATOR_COLUMNS: List[str] = [
    # price
    "sma_7", "sma_14", "sma_30",
    "ema_7", "ema_12", "ema_26",
    "rsi_14",
    "macd", "macd_hist", "macd_signal",
    "bb_lower", "bb_middle", "bb_upper", "bb_width", "bb_position",
    # range
    "hl_spread", "hl_spread_pct", "close_to_high", "close_to_low",
    # volume
    "volume_sma_7", "volume_sma_14", "volume_roc", "volume_ratio_7", "obv",
    # returns
    "return_1d", "return_7d", "return_14d", "log_return_1d",
    "volatility_7d", "volatility_14d", "volatility_30d",
    # momentum
    "roc_7", "roc_14", "momentum_7", "momentum_14",
]
