#!/usr/bin/env python3
"""Scheduler daemon for automated data collection and model training"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger

from src.config import settings
from src.collectors import MarketCollector
from src.features import FeatureEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Scheduled tasks
async def collect_market_data():
    """Collect market order snapshots (every 5 minutes)"""
    logger.info("Running scheduled market data collection...")
    collector = MarketCollector()
    await collector.collect_order_snapshots()


async def compute_features():
    """Compute features for all tracked items (every hour)"""
    logger.info("Running scheduled feature computation...")
    from src.collectors.market_collector import MarketCollector

    feature_engine = FeatureEngine()

    for type_id in MarketCollector.DEFAULT_TYPE_IDS:
        await feature_engine.compute_and_store_features(
            type_id=type_id,
            region_id=MarketCollector.DEFAULT_REGION_ID,
        )


async def retrain_model():
    """Retrain model (weekly)"""
    logger.info("Running scheduled model retraining...")
    # Model retraining logic would go here
    # This is a placeholder
    pass


async def main():
    """Run scheduler daemon"""
    logger.info("Starting scheduler daemon...")

    scheduler = AsyncIOScheduler()

    # Market data collection (every 5 minutes)
    scheduler.add_job(
        collect_market_data,
        IntervalTrigger(minutes=settings.collection_interval_minutes),
        id="market_collector",
        name="Market Data Collection",
    )

    # Feature computation (every hour)
    scheduler.add_job(
        compute_features,
        IntervalTrigger(hours=settings.feature_computation_interval_hours),
        id="feature_computer",
        name="Feature Computation",
    )

    # Model retraining (weekly)
    # APScheduler day_of_week: mon,tue,wed,thu,fri,sat,sun or 0-6
    scheduler.add_job(
        retrain_model,
        CronTrigger(
            day_of_week=settings.retraining_day,  # Use "sun" not "sunday"
            hour=settings.retraining_hour,
        ),
        id="model_retrainer",
        name="Weekly Model Retraining",
    )

    # Start scheduler
    scheduler.start()
    logger.info("Scheduler started. Press Ctrl+C to exit.")

    try:
        # Keep running
        await asyncio.Event().wait()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Shutting down scheduler...")
        scheduler.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
