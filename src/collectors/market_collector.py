"""Market data collector service"""

import logging
from datetime import datetime, timezone
from typing import List
from sqlalchemy import select

from src.config import settings
from src.collectors.esi_client import ESIClient
from src.database import get_session, MarketHistory, OrderSnapshot, Item

logger = logging.getLogger(__name__)


class MarketCollector:
    """Collects market data from ESI API and stores in database"""

    # Default items to track (high-liquidity items in Jita)
    DEFAULT_TYPE_IDS = [
        44992,   # PLEX
        40519,   # Large Skill Injector
        40520,   # Large Skill Extractor
        34,      # Tritanium
        35,      # Pyerite
        36,      # Mexallon
        37,      # Isogen
        38,      # Nocxium
        39,      # Zydrine
        40,      # Megacyte
        11399,   # Mercoxit
    ]

    # The Forge (Jita)
    DEFAULT_REGION_ID = 10000002

    def __init__(
        self,
        region_id: int = DEFAULT_REGION_ID,
        type_ids: List[int] = None,
    ):
        self.region_id = region_id
        self.type_ids = type_ids or self.DEFAULT_TYPE_IDS

    async def collect_historical_data(self):
        """
        Collect historical market data for all tracked items

        This fetches the last 365 days of daily statistics
        """
        logger.info(
            f"Collecting historical data for {len(self.type_ids)} items "
            f"in region {self.region_id}"
        )

        async with ESIClient() as client:
            # Bulk fetch all historical data with rate limiting
            history_data = await client.bulk_get_market_history(
                self.region_id, self.type_ids
            )

            logger.info(f"Received history for {len(history_data)} items")

            # Store in database
            async with get_session() as session:
                total_records = 0

                for type_id, history in history_data.items():
                    if not history:
                        continue

                    records = []
                    for day_data in history:
                        # Convert date string to datetime (naive, no timezone)
                        date_str = day_data.get("date")
                        if not date_str:
                            continue

                        dt = datetime.fromisoformat(date_str)

                        record = MarketHistory(
                            time=dt,
                            region_id=self.region_id,
                            type_id=type_id,
                            average=day_data.get("average"),
                            highest=day_data.get("highest"),
                            lowest=day_data.get("lowest"),
                            volume=day_data.get("volume"),
                            order_count=day_data.get("order_count"),
                        )
                        records.append(record)

                    # Upsert records (insert or update)
                    for record in records:
                        await session.merge(record)

                    total_records += len(records)

                    if total_records % 1000 == 0:
                        logger.info(f"Stored {total_records} historical records...")

                await session.commit()
                logger.info(f"Stored total of {total_records} historical records")

    async def collect_order_snapshots(self):
        """
        Collect current order book snapshot for all tracked items

        Should be run every 5 minutes to match ESI cache
        """
        snapshot_time = datetime.utcnow()  # Naive UTC datetime

        logger.info(
            f"Collecting order snapshot at {snapshot_time} for "
            f"{len(self.type_ids)} items"
        )

        async with ESIClient() as client:
            # Bulk fetch all order data
            orders_data = await client.bulk_get_market_orders(
                self.region_id, self.type_ids
            )

            logger.info(f"Received orders for {len(orders_data)} items")

            # Store in database
            async with get_session() as session:
                total_orders = 0

                for type_id, orders in orders_data.items():
                    if not orders:
                        continue

                    for order in orders:
                        # Parse issued datetime (naive, no timezone)
                        issued_str = order.get("issued")
                        issued_dt = None
                        if issued_str:
                            # Remove timezone indicator and parse as naive datetime
                            # ESI returns ISO format with 'Z' suffix like '2025-10-25T02:02:59Z'
                            clean_str = issued_str.replace("Z", "")
                            # Also handle +00:00 format if present
                            if "+00:00" in clean_str:
                                clean_str = clean_str.replace("+00:00", "")
                            issued_dt = datetime.fromisoformat(clean_str)

                        record = OrderSnapshot(
                            snapshot_time=snapshot_time,
                            order_id=order["order_id"],
                            type_id=type_id,
                            region_id=self.region_id,
                            location_id=order["location_id"],
                            price=order["price"],
                            volume_remain=order["volume_remain"],
                            volume_total=order["volume_total"],
                            is_buy_order=order["is_buy_order"],
                            duration=order.get("duration"),
                            issued=issued_dt,
                            min_volume=order.get("min_volume", 1),
                            range=order.get("range"),
                        )

                        session.add(record)
                        total_orders += 1

                await session.commit()
                logger.info(f"Stored {total_orders} order snapshots")

    async def update_item_metadata(self):
        """Fetch and store item metadata (names, categories)"""
        logger.info("Updating item metadata...")

        async with ESIClient() as client:
            async with get_session() as session:
                for type_id in self.type_ids:
                    # Check if already in database
                    result = await session.execute(
                        select(Item).where(Item.type_id == type_id)
                    )
                    existing = result.scalar_one_or_none()

                    if existing:
                        logger.debug(f"Item {type_id} metadata already exists")
                        continue

                    # Fetch from ESI
                    info = await client.get_item_info(type_id)
                    if not info:
                        logger.warning(f"Could not fetch info for type_id {type_id}")
                        continue

                    item = Item(
                        type_id=type_id,
                        name=info.get("name", "Unknown"),
                        category=str(info.get("group_id")) if info.get("group_id") else None,
                        is_tradeable=info.get("published", True),
                    )

                    session.add(item)

                await session.commit()
                logger.info(f"Updated metadata for {len(self.type_ids)} items")

    async def calculate_liquidity_tiers(self):
        """
        Calculate liquidity tiers based on recent trading volume

        Classifies items as high/medium/low liquidity
        """
        logger.info("Calculating liquidity tiers...")

        async with get_session() as session:
            # Get average daily volume for each item (last 30 days)
            from sqlalchemy import func
            from datetime import timedelta

            thirty_days_ago = datetime.utcnow() - timedelta(days=30)

            for type_id in self.type_ids:
                result = await session.execute(
                    select(
                        func.avg(MarketHistory.volume).label("avg_volume"),
                        func.avg(MarketHistory.order_count).label("avg_orders"),
                    )
                    .where(MarketHistory.type_id == type_id)
                    .where(MarketHistory.region_id == self.region_id)
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

                # Update item record
                result = await session.execute(
                    select(Item).where(Item.type_id == type_id)
                )
                item = result.scalar_one_or_none()

                if item:
                    item.liquidity_tier = tier
                    item.avg_daily_volume = int(avg_volume)
                    item.avg_daily_orders = int(avg_orders)

            await session.commit()
            logger.info("Liquidity tiers updated")
