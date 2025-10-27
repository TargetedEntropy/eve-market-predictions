"""Technical indicators using pandas-ta"""

import pandas as pd
import pandas_ta as ta
import numpy as np
from typing import Dict


class TechnicalIndicators:
    """Compute technical indicators for price series"""

    @staticmethod
    def compute_price_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute price-based technical indicators

        Args:
            df: DataFrame with 'close' column (or 'average' for market history)

        Returns:
            DataFrame with additional indicator columns
        """
        # Use 'average' if available, otherwise 'close'
        price_col = "average" if "average" in df.columns else "close"
        prices = df[price_col]

        # Simple Moving Averages
        df["sma_7"] = ta.sma(prices, length=7)
        df["sma_14"] = ta.sma(prices, length=14)
        df["sma_30"] = ta.sma(prices, length=30)

        # Exponential Moving Averages
        df["ema_7"] = ta.ema(prices, length=7)
        df["ema_12"] = ta.ema(prices, length=12)
        df["ema_26"] = ta.ema(prices, length=26)

        # Relative Strength Index
        df["rsi_14"] = ta.rsi(prices, length=14)

        # MACD
        macd = ta.macd(prices, fast=12, slow=26, signal=9)
        df = pd.concat([df, macd], axis=1)

        # Bollinger Bands
        bbands = ta.bbands(prices, length=20, std=2)
        df = pd.concat([df, bbands], axis=1)

        # Average True Range (volatility)
        if "high" in df.columns and "low" in df.columns:
            df["atr_14"] = ta.atr(df["high"], df["low"], prices, length=14)

        return df

    @staticmethod
    def compute_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute volume-based indicators

        Args:
            df: DataFrame with 'volume' column

        Returns:
            DataFrame with volume indicators
        """
        if "volume" not in df.columns:
            return df

        volumes = df["volume"]

        # Volume moving averages
        df["volume_sma_7"] = ta.sma(volumes, length=7)
        df["volume_sma_14"] = ta.sma(volumes, length=14)

        # Volume rate of change
        df["volume_roc"] = volumes.pct_change(periods=1)

        return df

    @staticmethod
    def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute price returns and related features

        Args:
            df: DataFrame with price column

        Returns:
            DataFrame with return features
        """
        price_col = "average" if "average" in df.columns else "close"
        prices = df[price_col]

        # Simple returns
        df["return_1d"] = prices.pct_change(1)
        df["return_7d"] = prices.pct_change(7)
        df["return_14d"] = prices.pct_change(14)

        # Log returns (better for ML)
        df["log_return_1d"] = np.log(prices / prices.shift(1))

        # Rolling volatility (std of returns)
        df["volatility_7d"] = df["return_1d"].rolling(window=7).std()
        df["volatility_14d"] = df["return_1d"].rolling(window=14).std()
        df["volatility_30d"] = df["return_1d"].rolling(window=30).std()

        return df

    @staticmethod
    def compute_momentum(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute momentum indicators

        Args:
            df: DataFrame with price column

        Returns:
            DataFrame with momentum features
        """
        price_col = "average" if "average" in df.columns else "close"
        prices = df[price_col]

        # Rate of change
        df["roc_7"] = ta.roc(prices, length=7)
        df["roc_14"] = ta.roc(prices, length=14)

        # Momentum
        df["momentum_7"] = ta.mom(prices, length=7)
        df["momentum_14"] = ta.mom(prices, length=14)

        return df

    @staticmethod
    def compute_all(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all technical indicators

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with all indicators
        """
        df = TechnicalIndicators.compute_price_indicators(df)
        df = TechnicalIndicators.compute_volume_indicators(df)
        df = TechnicalIndicators.compute_returns(df)
        df = TechnicalIndicators.compute_momentum(df)

        return df
