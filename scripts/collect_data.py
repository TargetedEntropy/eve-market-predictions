#!/usr/bin/env python3
"""One-off data collection.

The scheduler daemon (``scripts/scheduler_daemon.py``) runs all of these on a
schedule; this script is for bootstrapping a fresh database or backfilling by
hand.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.collectors import MarketCollector
from src.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region-id", type=int, action="append", dest="region_ids",
                        help="Region to collect (repeatable; default: COLLECTION_REGION_IDS)")
    parser.add_argument("--skip-history", action="store_true")
    parser.add_argument("--skip-orders", action="store_true")
    parser.add_argument("--skip-metadata", action="store_true")
    parser.add_argument("--skip-reference-prices", action="store_true")
    return parser.parse_args(argv)


async def main(argv=None):
    args = parse_args(argv)
    collector = MarketCollector(region_ids=args.region_ids or settings.region_ids)

    logger.info(f"Collecting for regions {collector.region_ids}")

    if not args.skip_metadata:
        await collector.update_item_metadata()

    if not args.skip_history:
        await collector.collect_historical_data()

    if not args.skip_orders:
        await collector.collect_order_snapshots()

    if not args.skip_reference_prices:
        await collector.collect_reference_prices()

    if not args.skip_history:
        await collector.calculate_liquidity_tiers()

    logger.info("Data collection complete!")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
