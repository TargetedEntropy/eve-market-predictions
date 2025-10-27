"""Feature engineering pipeline"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from sqlalchemy import select

from src.config import settings
from src.database import get_session, MarketHistory, OrderSnapshot, Feature
from src.features.technical_indicators import TechnicalIndicators

logger = logging.getLogger(__name__)


class FeatureEngine:
    """Compute and store features for ML models"""

    @staticmethod
    def compute_order_book_features(orders_df: pd.DataFrame) -> Dict[str, float]:
        """
        Compute features from order book snapshot

        Args:
            orders_df: DataFrame of orders with columns:
                       price, volume_remain, is_buy_order

        Returns:
            Dictionary of order book features
        """
        if orders_df.empty:
            return {}

        features = {}

        # Separate buy and sell orders
        buy_orders = orders_df[orders_df["is_buy_order"] == True]
        sell_orders = orders_df[orders_df["is_buy_order"] == False]

        # Best bid/ask
        if not buy_orders.empty:
            features["best_bid"] = float(buy_orders["price"].max())
            features["bid_volume"] = float(buy_orders["volume_remain"].sum())
        else:
            features["best_bid"] = 0.0
            features["bid_volume"] = 0.0

        if not sell_orders.empty:
            features["best_ask"] = float(sell_orders["price"].min())
            features["ask_volume"] = float(sell_orders["volume_remain"].sum())
        else:
            features["best_ask"] = 0.0
            features["ask_volume"] = 0.0

        # Spread
        if features["best_bid"] > 0 and features["best_ask"] > 0:
            features["spread_abs"] = features["best_ask"] - features["best_bid"]
            features["spread_pct"] = (
                features["spread_abs"] / features["best_ask"]
            ) * 100
            features["mid_price"] = (features["best_bid"] + features["best_ask"]) / 2
        else:
            features["spread_abs"] = 0.0
            features["spread_pct"] = 0.0
            features["mid_price"] = 0.0

        # Order book imbalance
        total_volume = features["bid_volume"] + features["ask_volume"]
        if total_volume > 0:
            features["imbalance"] = (
                features["bid_volume"] - features["ask_volume"]
            ) / total_volume
        else:
            features["imbalance"] = 0.0

        # Order depth (top 5 levels)
        if not buy_orders.empty:
            top_5_bids = buy_orders.nlargest(5, "price")
            features["bid_depth_5"] = float(top_5_bids["volume_remain"].sum())
        else:
            features["bid_depth_5"] = 0.0

        if not sell_orders.empty:
            top_5_asks = sell_orders.nsmallest(5, "price")
            features["ask_depth_5"] = float(top_5_asks["volume_remain"].sum())
        else:
            features["ask_depth_5"] = 0.0

        # Total order counts
        features["total_orders"] = len(orders_df)
        features["buy_order_count"] = len(buy_orders)
        features["sell_order_count"] = len(sell_orders)

        return features

    @staticmethod
    def compute_temporal_features(dt: datetime) -> Dict[str, float]:
        """
        Compute temporal features with cyclic encoding

        Args:
            dt: Datetime to extract features from

        Returns:
            Dictionary of temporal features
        """
        features = {}

        # Hour (cyclic encoding)
        hour = dt.hour
        features["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        features["hour_cos"] = np.cos(2 * np.pi * hour / 24)

        # Day of week (cyclic encoding)
        day_of_week = dt.weekday()
        features["dow_sin"] = np.sin(2 * np.pi * day_of_week / 7)
        features["dow_cos"] = np.cos(2 * np.pi * day_of_week / 7)

        # Day of month (cyclic encoding)
        day_of_month = dt.day
        features["dom_sin"] = np.sin(2 * np.pi * day_of_month / 30)
        features["dom_cos"] = np.cos(2 * np.pi * day_of_month / 30)

        # Binary features
        features["is_weekend"] = float(day_of_week >= 5)
        features["is_peak_hour"] = float(18 <= hour <= 22)  # EVE prime time

        return features

    async def compute_and_store_features(
        self,
        type_id: int,
        region_id: int,
        target_time: Optional[datetime] = None,
    ):
        """
        Compute all features and store in database

        Args:
            type_id: Item type ID
            region_id: Region ID
            target_time: Time to compute features for (default: now)
        """
        if target_time is None:
            target_time = datetime.now(timezone.utc)

        logger.info(f"Computing features for type_id={type_id} at {target_time}")

        async with get_session() as session:
            # Fetch historical price data (last 90 days)
            ninety_days_ago = target_time - timedelta(days=90)

            result = await session.execute(
                select(MarketHistory)
                .where(MarketHistory.type_id == type_id)
                .where(MarketHistory.region_id == region_id)
                .where(MarketHistory.time >= ninety_days_ago)
                .where(MarketHistory.time <= target_time)
                .order_by(MarketHistory.time)
            )

            history_records = result.scalars().all()

            if not history_records:
                logger.warning(f"No historical data for type_id={type_id}")
                return

            # Convert to DataFrame
            history_df = pd.DataFrame(
                [
                    {
                        "time": r.time,
                        "average": float(r.average) if r.average else None,
                        "highest": float(r.highest) if r.highest else None,
                        "lowest": float(r.lowest) if r.lowest else None,
                        "volume": r.volume,
                        "order_count": r.order_count,
                    }
                    for r in history_records
                ]
            )

            # Compute technical indicators
            history_df = TechnicalIndicators.compute_all(history_df)

            # Get latest values (most recent day)
            latest_features = history_df.iloc[-1].to_dict()

            # Fetch latest order snapshot
            result = await session.execute(
                select(OrderSnapshot)
                .where(OrderSnapshot.type_id == type_id)
                .where(OrderSnapshot.region_id == region_id)
                .where(OrderSnapshot.snapshot_time <= target_time)
                .order_by(OrderSnapshot.snapshot_time.desc())
                .limit(1000)  # Get one snapshot worth of orders
            )

            order_records = result.scalars().all()

            if order_records:
                orders_df = pd.DataFrame(
                    [
                        {
                            "price": float(r.price),
                            "volume_remain": r.volume_remain,
                            "is_buy_order": r.is_buy_order,
                        }
                        for r in order_records
                    ]
                )

                # Compute order book features
                order_features = self.compute_order_book_features(orders_df)
                latest_features.update(order_features)

            # Compute temporal features
            temporal_features = self.compute_temporal_features(target_time)
            latest_features.update(temporal_features)

            # Store features in database
            for feature_name, feature_value in latest_features.items():
                if pd.isna(feature_value) or feature_value is None:
                    continue

                feature = Feature(
                    time=target_time,
                    type_id=type_id,
                    region_id=region_id,
                    feature_name=feature_name,
                    feature_value=float(feature_value),
                )

                await session.merge(feature)

            await session.commit()
            logger.info(f"Stored {len(latest_features)} features for type_id={type_id}")

    async def get_feature_vector(
        self,
        type_id: int,
        region_id: int,
        target_time: Optional[datetime] = None,
    ) -> Optional[Dict[str, float]]:
        """
        Retrieve feature vector for a given time

        Args:
            type_id: Item type ID
            region_id: Region ID
            target_time: Time to get features for (default: now)

        Returns:
            Dictionary of features or None if not found
        """
        if target_time is None:
            target_time = datetime.now(timezone.utc)

        async with get_session() as session:
            result = await session.execute(
                select(Feature)
                .where(Feature.type_id == type_id)
                .where(Feature.region_id == region_id)
                .where(Feature.time == target_time)
            )

            feature_records = result.scalars().all()

            if not feature_records:
                return None

            return {
                r.feature_name: float(r.feature_value)
                for r in feature_records
                if r.feature_value is not None
            }
