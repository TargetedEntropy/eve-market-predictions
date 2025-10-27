"""Technical indicators using 'ta' library (alternative to pandas-ta)"""

import pandas as pd
import numpy as np
from typing import Dict

# Try to import pandas_ta, fall back to 'ta' library
try:
    import pandas_ta as ta
    USE_PANDAS_TA = True
except ImportError:
    try:
        import ta as talib
        USE_PANDAS_TA = False
        print("Using 'ta' library instead of pandas-ta")
    except ImportError:
        print("Warning: No technical analysis library found. Install with:")
        print("  pip install git+https://github.com/twopirllc/pandas-ta.git")
        print("  OR pip install ta")
        USE_PANDAS_TA = None


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

        if USE_PANDAS_TA is None:
            # No library available, compute basic indicators manually
            df["sma_7"] = prices.rolling(window=7).mean()
            df["sma_14"] = prices.rolling(window=14).mean()
            df["sma_30"] = prices.rolling(window=30).mean()
            df["ema_12"] = prices.ewm(span=12).mean()
            df["ema_26"] = prices.ewm(span=26).mean()
            return df

        if USE_PANDAS_TA:
            # Using pandas-ta
            df["sma_7"] = ta.sma(prices, length=7)
            df["sma_14"] = ta.sma(prices, length=14)
            df["sma_30"] = ta.sma(prices, length=30)
            df["ema_7"] = ta.ema(prices, length=7)
            df["ema_12"] = ta.ema(prices, length=12)
            df["ema_26"] = ta.ema(prices, length=26)
            df["rsi_14"] = ta.rsi(prices, length=14)

            # MACD
            macd = ta.macd(prices, fast=12, slow=26, signal=9)
            df = pd.concat([df, macd], axis=1)

            # Bollinger Bands
            bbands = ta.bbands(prices, length=20, std=2)
            df = pd.concat([df, bbands], axis=1)
        else:
            # Using 'ta' library
            df["sma_7"] = talib.trend.sma_indicator(prices, window=7)
            df["sma_14"] = talib.trend.sma_indicator(prices, window=14)
            df["sma_30"] = talib.trend.sma_indicator(prices, window=30)
            df["ema_12"] = talib.trend.ema_indicator(prices, window=12)
            df["ema_26"] = talib.trend.ema_indicator(prices, window=26)
            df["rsi_14"] = talib.momentum.rsi(prices, window=14)

            # MACD
            macd_obj = talib.trend.MACD(prices, window_slow=26, window_fast=12, window_sign=9)
            df["MACD_12_26_9"] = macd_obj.macd()
            df["MACDh_12_26_9"] = macd_obj.macd_diff()
            df["MACDs_12_26_9"] = macd_obj.macd_signal()

            # Bollinger Bands
            bollinger = talib.volatility.BollingerBands(prices, window=20, window_dev=2)
            df["BBL_20_2.0"] = bollinger.bollinger_lband()
            df["BBM_20_2.0"] = bollinger.bollinger_mavg()
            df["BBU_20_2.0"] = bollinger.bollinger_uband()

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

        # Volume moving averages (works without any library)
        df["volume_sma_7"] = volumes.rolling(window=7).mean()
        df["volume_sma_14"] = volumes.rolling(window=14).mean()
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

        # Simple momentum (works without library)
        df["roc_7"] = prices.pct_change(periods=7) * 100
        df["roc_14"] = prices.pct_change(periods=14) * 100
        df["momentum_7"] = prices.diff(periods=7)
        df["momentum_14"] = prices.diff(periods=14)

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
