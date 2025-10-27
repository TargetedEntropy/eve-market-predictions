#!/usr/bin/env python3
"""Data collection script"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.collectors import MarketCollector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Run data collection"""
    logger.info("Starting data collection...")

    collector = MarketCollector()

    # Update item metadata
    await collector.update_item_metadata()

    # Collect historical data
    await collector.collect_historical_data()

    # Collect order snapshot
    await collector.collect_order_snapshots()

    # Calculate liquidity tiers
    await collector.calculate_liquidity_tiers()

    logger.info("Data collection complete!")


if __name__ == "__main__":
    asyncio.run(main())
