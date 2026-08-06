"""Feature engineering pipeline"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sqlalchemy import select

from src.database import Feature, MarketHistory, OrderSnapshot, get_session
from src.database.upsert import bulk_upsert
from src.features.technical_indicators import TechnicalIndicators
from src.utils.timeutils import to_naive_utc, utcnow

logger = logging.getLogger(__name__)

#: Columns of the history frame that are inputs, not features to store.
NON_FEATURE_COLUMNS = {"time"}


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
        buy_orders = orders_df[orders_df["is_buy_order"]]
        sell_orders = orders_df[~orders_df["is_buy_order"]]

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
        features["total_orders"] = float(len(orders_df))
        features["buy_order_count"] = float(len(buy_orders))
        features["sell_order_count"] = float(len(sell_orders))

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
        features["hour_sin"] = float(np.sin(2 * np.pi * hour / 24))
        features["hour_cos"] = float(np.cos(2 * np.pi * hour / 24))

        # Day of week (cyclic encoding)
        day_of_week = dt.weekday()
        features["dow_sin"] = float(np.sin(2 * np.pi * day_of_week / 7))
        features["dow_cos"] = float(np.cos(2 * np.pi * day_of_week / 7))

        # Day of month (cyclic encoding)
        day_of_month = dt.day
        features["dom_sin"] = float(np.sin(2 * np.pi * day_of_month / 30))
        features["dom_cos"] = float(np.cos(2 * np.pi * day_of_month / 30))

        # Binary features
        features["is_weekend"] = float(day_of_week >= 5)
        features["is_peak_hour"] = float(18 <= hour <= 22)  # EVE prime time

        return features

    @staticmethod
    def _numeric_features(row: Dict[str, object]) -> Dict[str, float]:
        """Keep only finite numeric values.

        ``history_df.iloc[-1].to_dict()`` includes the ``time`` column (a
        ``pd.Timestamp``); ``float()`` on that raises ``TypeError`` and used to
        abort the whole hourly job.
        """
        numeric: Dict[str, float] = {}
        for name, value in row.items():
            if name in NON_FEATURE_COLUMNS:
                continue
            if isinstance(value, (bool, np.bool_)):
                numeric[name] = float(value)
                continue
            if isinstance(value, (pd.Timestamp, datetime, str, bytes)) or value is None:
                continue
            try:
                as_float = float(value)
            except (TypeError, ValueError):
                continue
            if np.isnan(as_float) or np.isinf(as_float):
                continue
            numeric[name] = as_float
        return numeric

    async def compute_and_store_features(
        self,
        type_id: int,
        region_id: int,
        target_time: Optional[datetime] = None,
    ) -> int:
        """
        Compute all features and store in database

        Args:
            type_id: Item type ID
            region_id: Region ID
            target_time: Naive UTC time to compute features for (default: now)

        Returns:
            Number of features stored.
        """
        target_time = to_naive_utc(target_time) if target_time else utcnow()

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
                return 0

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

            history_df = TechnicalIndicators.compute_all(history_df)

            # Most recent day, numeric columns only
            latest_features = self._numeric_features(history_df.iloc[-1].to_dict())

            order_features = await self._latest_order_book_features(
                session, type_id, region_id, target_time
            )
            latest_features.update(order_features)

            latest_features.update(self.compute_temporal_features(target_time))

            rows = [
                {
                    "time": target_time,
                    "type_id": type_id,
                    "region_id": region_id,
                    "feature_name": name,
                    "feature_value": value,
                }
                for name, value in latest_features.items()
            ]

            written = await bulk_upsert(session, Feature, rows)
            await session.commit()

        logger.info(f"Stored {written} features for type_id={type_id}")
        return written

    async def _latest_order_book_features(
        self,
        session,
        type_id: int,
        region_id: int,
        target_time: datetime,
    ) -> Dict[str, float]:
        """Order-book features from the most recent complete snapshot.

        Selecting "the newest 1000 rows" could straddle two snapshots; this
        pins to one ``snapshot_time`` and reads all of its orders.
        """
        latest_snapshot = (
            await session.execute(
                select(OrderSnapshot.snapshot_time)
                .where(OrderSnapshot.type_id == type_id)
                .where(OrderSnapshot.region_id == region_id)
                .where(OrderSnapshot.snapshot_time <= target_time)
                .order_by(OrderSnapshot.snapshot_time.desc())
                .limit(1)
            )
        ).scalar_one_or_none()

        if latest_snapshot is None:
            return {}

        order_records = (
            await session.execute(
                select(OrderSnapshot)
                .where(OrderSnapshot.type_id == type_id)
                .where(OrderSnapshot.region_id == region_id)
                .where(OrderSnapshot.snapshot_time == latest_snapshot)
            )
        ).scalars().all()

        if not order_records:
            return {}

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

        return self.compute_order_book_features(orders_df)

    async def get_feature_vector(
        self,
        type_id: int,
        region_id: int,
        target_time: Optional[datetime] = None,
        max_age: Optional[timedelta] = None,
    ) -> Optional[Dict[str, float]]:
        """
        Retrieve the most recent feature vector at or before ``target_time``.

        Features are written at scheduler tick times, so an exact equality
        against "now" (the old behaviour) never matched and every ``/predict``
        call 404'd.

        Args:
            type_id: Item type ID
            region_id: Region ID
            target_time: Naive UTC upper bound (default: now)
            max_age: Reject feature sets older than this

        Returns:
            Dictionary of features or None if nothing suitable exists
        """
        target_time = to_naive_utc(target_time) if target_time else utcnow()

        async with get_session() as session:
            latest_time = (
                await session.execute(
                    select(Feature.time)
                    .where(Feature.type_id == type_id)
                    .where(Feature.region_id == region_id)
                    .where(Feature.time <= target_time)
                    .order_by(Feature.time.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()

            if latest_time is None:
                return None

            if max_age is not None and target_time - latest_time > max_age:
                logger.warning(
                    f"Newest features for type_id={type_id} are from {latest_time}, "
                    f"older than the {max_age} freshness limit"
                )
                return None

            feature_records = (
                await session.execute(
                    select(Feature)
                    .where(Feature.type_id == type_id)
                    .where(Feature.region_id == region_id)
                    .where(Feature.time == latest_time)
                )
            ).scalars().all()

            return {
                r.feature_name: float(r.feature_value)
                for r in feature_records
                if r.feature_value is not None
            }
