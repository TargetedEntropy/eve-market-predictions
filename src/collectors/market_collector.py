"""Market data collector service"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from sqlalchemy import func, select

from src.collectors.esi_client import ESIClient
from src.config import settings
from src.constants import DEFAULT_REGION_ID, TRACKED_TYPE_IDS
from src.database import Item, MarketHistory, MarketPrice, OrderSnapshot, get_session
from src.database.upsert import bulk_upsert
from src.utils.timeutils import parse_esi_datetime, utcnow

logger = logging.getLogger(__name__)


class MarketCollector:
    """Collects market data from ESI API and stores in database.

    The tracked item list lives in :mod:`src.constants` so the collector,
    dashboard, drift detector and prediction scripts cannot disagree about
    which type_id maps to which item.
    """

    DEFAULT_TYPE_IDS = TRACKED_TYPE_IDS
    DEFAULT_REGION_ID = DEFAULT_REGION_ID

    def __init__(
        self,
        region_ids: Optional[List[int]] = None,
        type_ids: Optional[List[int]] = None,
    ):
        self.region_ids = list(region_ids or settings.region_ids)
        self.type_ids = list(type_ids or TRACKED_TYPE_IDS)

    @property
    def region_id(self) -> int:
        """Primary region (first configured)."""
        return self.region_ids[0]

    async def collect_historical_data(self):
        """
        Collect historical market data for all tracked items and regions.

        ESI serves roughly the last year of daily statistics and refreshes it
        once a day, so the scheduler runs this daily.
        """
        logger.info(
            f"Collecting historical data for {len(self.type_ids)} items "
            f"in regions {self.region_ids}"
        )

        total_records = 0

        async with ESIClient() as client:
            for region_id in self.region_ids:
                history_data = await client.bulk_get_market_history(region_id, self.type_ids)
                logger.info(f"Received history for {len(history_data)} items in {region_id}")

                rows = []
                for type_id, history in history_data.items():
                    for day_data in history or []:
                        date_str = day_data.get("date")
                        if not date_str:
                            continue
                        rows.append(
                            {
                                "time": datetime.fromisoformat(date_str),
                                "region_id": region_id,
                                "type_id": type_id,
                                "average": day_data.get("average"),
                                "highest": day_data.get("highest"),
                                "lowest": day_data.get("lowest"),
                                "volume": day_data.get("volume"),
                                "order_count": day_data.get("order_count"),
                            }
                        )

                if not rows:
                    continue

                async with get_session() as session:
                    written = await bulk_upsert(session, MarketHistory, rows)
                    await session.commit()

                total_records += written
                logger.info(f"Stored {written} history rows for region {region_id}")

            logger.info(f"ESI client stats: {client.get_stats()}")

        logger.info(f"Stored total of {total_records} historical records")
        return total_records

    async def collect_order_snapshots(self):
        """
        Collect current order book snapshot for all tracked items

        Should be run every 5 minutes to match ESI cache
        """
        snapshot_time = utcnow()

        logger.info(
            f"Collecting order snapshot at {snapshot_time} for "
            f"{len(self.type_ids)} items in regions {self.region_ids}"
        )

        total_orders = 0

        async with ESIClient() as client:
            for region_id in self.region_ids:
                orders_data = await client.bulk_get_market_orders(region_id, self.type_ids)
                logger.info(f"Received orders for {len(orders_data)} items in {region_id}")

                rows = []
                for type_id, orders in orders_data.items():
                    for order in orders or []:
                        rows.append(
                            {
                                "snapshot_time": snapshot_time,
                                "order_id": order["order_id"],
                                "type_id": type_id,
                                "region_id": region_id,
                                "location_id": order["location_id"],
                                "price": order["price"],
                                "volume_remain": order["volume_remain"],
                                "volume_total": order["volume_total"],
                                "is_buy_order": order["is_buy_order"],
                                "duration": order.get("duration"),
                                "issued": parse_esi_datetime(order.get("issued")),
                                "min_volume": order.get("min_volume", 1),
                                "range": order.get("range"),
                            }
                        )

                if not rows:
                    continue

                # Upsert rather than plain insert: (snapshot_time, order_id) is
                # the primary key, so a retried run updates instead of failing.
                async with get_session() as session:
                    written = await bulk_upsert(session, OrderSnapshot, rows)
                    await session.commit()

                total_orders += written

            logger.info(f"ESI client stats: {client.get_stats()}")

        logger.info(f"Stored {total_orders} order snapshots")
        return total_orders

    async def collect_reference_prices(self):
        """Store universe-wide adjusted/average prices from ``/markets/prices/``.

        These are CCP's own reference values, used for industry cost indices,
        and are a useful sanity check against regional market prices.
        """
        collected_at = utcnow()
        logger.info("Collecting ESI reference prices...")

        async with ESIClient() as client:
            prices = await client.get_market_prices()

        tracked = set(self.type_ids)
        rows = [
            {
                "time": collected_at,
                "type_id": entry["type_id"],
                "adjusted_price": entry.get("adjusted_price"),
                "average_price": entry.get("average_price"),
            }
            for entry in prices
            if entry.get("type_id") in tracked
        ]

        if not rows:
            logger.warning("No reference prices returned for tracked items")
            return 0

        async with get_session() as session:
            written = await bulk_upsert(session, MarketPrice, rows)
            await session.commit()

        logger.info(f"Stored {written} reference prices")
        return written

    async def update_item_metadata(self):
        """Fetch and refresh item metadata (names, group and category names).

        Unlike the previous skip-if-exists version this always refreshes, so
        renames and newly resolved categories propagate.
        """
        logger.info("Updating item metadata...")

        group_cache: Dict[int, Dict[str, Optional[str]]] = {}
        rows = []

        async with ESIClient() as client:
            for type_id in self.type_ids:
                info = await client.get_item_info(type_id)
                if not info:
                    logger.warning(f"Could not fetch info for type_id {type_id}")
                    continue

                group_id = info.get("group_id")
                group_info = await self._resolve_group(client, group_id, group_cache)

                rows.append(
                    {
                        "type_id": type_id,
                        "name": info.get("name", "Unknown"),
                        "group_id": group_id,
                        "group_name": group_info.get("group_name"),
                        "category": group_info.get("category_name"),
                        "is_tradeable": info.get("published", True),
                        "updated_at": utcnow(),
                    }
                )

        if not rows:
            return 0

        async with get_session() as session:
            written = await bulk_upsert(session, Item, rows)
            await session.commit()

        logger.info(f"Updated metadata for {written} items")
        return written

    @staticmethod
    async def _resolve_group(
        client: ESIClient,
        group_id: Optional[int],
        cache: Dict[int, Dict[str, Optional[str]]],
    ) -> Dict[str, Optional[str]]:
        """Resolve group_id -> (group name, category name), memoized per run."""
        if not group_id:
            return {"group_name": None, "category_name": None}
        if group_id in cache:
            return cache[group_id]

        resolved: Dict[str, Optional[str]] = {"group_name": None, "category_name": None}
        group = await client.get_group_info(group_id)
        if group:
            resolved["group_name"] = group.get("name")
            category_id = group.get("category_id")
            if category_id:
                category = await client.get_category_info(category_id)
                if category:
                    resolved["category_name"] = category.get("name")

        cache[group_id] = resolved
        return resolved

    async def calculate_liquidity_tiers(self):
        """
        Calculate liquidity tiers based on recent trading volume

        Classifies items as high/medium/low liquidity
        """
        logger.info("Calculating liquidity tiers...")

        thirty_days_ago = utcnow() - timedelta(days=30)

        async with get_session() as session:
            for type_id in self.type_ids:
                result = await session.execute(
                    select(
                        func.avg(MarketHistory.volume).label("avg_volume"),
                        func.avg(MarketHistory.order_count).label("avg_orders"),
                    )
                    .where(MarketHistory.type_id == type_id)
                    .where(MarketHistory.region_id.in_(self.region_ids))
                    .where(MarketHistory.time >= thirty_days_ago)
                )

                stats = result.one_or_none()
                if not stats or not stats.avg_volume:
                    continue

                avg_volume = float(stats.avg_volume)
                avg_orders = float(stats.avg_orders) if stats.avg_orders else 0

                # Classify liquidity
                if avg_volume > 1_000_000_000 and avg_orders > 100:
                    tier = "high"
                elif avg_volume > 100_000_000 and avg_orders > 20:
                    tier = "medium"
                else:
                    tier = "low"

                item = (
                    await session.execute(select(Item).where(Item.type_id == type_id))
                ).scalar_one_or_none()

                if item:
                    item.liquidity_tier = tier
                    item.avg_daily_volume = int(avg_volume)
                    item.avg_daily_orders = int(avg_orders)

            await session.commit()
            logger.info("Liquidity tiers updated")
