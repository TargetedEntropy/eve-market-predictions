#!/usr/bin/env python3
"""Scheduler daemon for automated data collection, training and monitoring.

Jobs (all UTC, all ``max_instances=1`` + ``coalesce=True`` so a slow ESI
response cannot stack runs on top of each other):

===============================  =================================================
Interval / cron                  Job
===============================  =================================================
every COLLECTION_INTERVAL        order book snapshots
every FEATURE_INTERVAL           feature computation
daily HISTORY_COLLECTION_HOUR    market history + item metadata + liquidity tiers
daily HISTORY_HOUR + 1           reference prices, then prediction scoring
daily PREDICTION_HOUR            generate and store predictions for trained models
daily DRIFT_CHECK_HOUR           drift detection (retrains when RETRAIN_ON_DRIFT)
weekly RETRAINING_DAY/HOUR       retrain every item, promoting only better models
===============================  =================================================
"""

import asyncio
import functools
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from apscheduler.events import EVENT_JOB_ERROR
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from src.collectors import MarketCollector
from src.config import settings
from src.constants import TRACKED_TYPE_IDS, item_name
from src.features import FeatureEngine
from src.models.pipeline import (
    InsufficientData,
    ModelNotTrained,
    generate_predictions,
    train_item,
    trained_items,
)
from src.monitoring import DriftDetector
from src.monitoring.alerting import alert_job_failure, send_alert
from src.monitoring.evaluation import run_evaluation

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# A run that overruns by more than this is dropped rather than queued
MISFIRE_GRACE_SECONDS = 300


def job(name: str):
    """Wrap a scheduled coroutine so a failure alerts instead of vanishing."""

    def decorator(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            logger.info(f"Job start: {name}")
            try:
                result = await fn(*args, **kwargs)
            except Exception as e:
                logger.exception(f"Job failed: {name}")
                await alert_job_failure(name, e)
                raise
            logger.info(f"Job done: {name}")
            return result

        return wrapper

    return decorator


# --------------------------------------------------------------------------- #
# Jobs
# --------------------------------------------------------------------------- #

@job("order snapshots")
async def collect_market_data():
    """Collect market order snapshots"""
    await MarketCollector().collect_order_snapshots()


@job("market history")
async def collect_market_history():
    """Daily history refresh plus item metadata and liquidity tiers.

    ``market_history`` feeds training, drift detection, the dashboard and every
    prediction path -- without this job it only updated when somebody ran
    ``scripts/collect_data.py`` by hand.
    """
    collector = MarketCollector()
    await collector.collect_historical_data()
    await collector.update_item_metadata()
    await collector.calculate_liquidity_tiers()


@job("reference prices")
async def collect_reference_prices():
    """Store ESI's adjusted/average reference prices"""
    await MarketCollector().collect_reference_prices()


@job("feature computation")
async def compute_features():
    """Compute features for all tracked items in all configured regions"""
    feature_engine = FeatureEngine()

    for region_id in settings.region_ids:
        for type_id in TRACKED_TYPE_IDS:
            try:
                await feature_engine.compute_and_store_features(
                    type_id=type_id, region_id=region_id
                )
            except Exception as e:
                # One bad item must not stop the other ten.
                logger.exception(f"Feature computation failed for {type_id}: {e}")


@job("prediction generation")
async def generate_all_predictions():
    """Forecast from saved models and persist rows in ``predictions``"""
    models = trained_items()
    if not models:
        logger.warning("No trained models; skipping prediction generation")
        return

    for type_id, region_id in models:
        try:
            result = await generate_predictions(type_id, region_id)
            logger.info(
                f"{result['item_name']}: {result['current_price']:,.0f} -> "
                f"{result['predicted_price']:,.0f} ISK "
                f"({result['percent_change']:+.2f}% over {result['horizon_days']}d)"
            )
        except (ModelNotTrained, InsufficientData) as e:
            logger.warning(f"Skipping predictions for {type_id}: {e}")
            continue
        except Exception as e:
            logger.exception(f"Prediction failed for {type_id}: {e}")


@job("prediction scoring")
async def evaluate_predictions():
    """Backfill realized prices and refresh model performance metrics"""
    summary = await run_evaluation()
    logger.info(
        f"Scored {summary['scored_predictions']} predictions, "
        f"wrote {summary['performance_rows']} performance rows"
    )


@job("drift detection")
async def check_drift():
    """Daily drift check; optionally triggers retraining for drifted items"""
    detector = DriftDetector(reference_window_days=30, current_window_days=7)

    items = [
        (type_id, region_id)
        for region_id in settings.region_ids
        for type_id in TRACKED_TYPE_IDS
    ]
    reports = await detector.check_all_items(items, save_reports=True)
    await detector.save_drift_summary(reports, "data/drift_summary.json")

    drifted = [r for r in reports if r.data_drift_detected or r.prediction_drift_detected]
    if not drifted:
        logger.info("No drift detected")
        return

    names = ", ".join(f"{item_name(r.type_id)}({r.type_id})" for r in drifted)
    await send_alert(
        title="Market drift detected",
        message=f"{len(drifted)} of {len(reports)} tracked items drifted: {names}",
        severity="warning",
        context={"drifted": len(drifted), "checked": len(reports)},
    )

    if settings.retrain_on_drift:
        logger.info("RETRAIN_ON_DRIFT is set; retraining drifted items now")
        await retrain_models([(r.type_id, r.region_id) for r in drifted])


@job("model retraining")
async def retrain_models(targets=None):
    """Retrain models, promoting a new checkpoint only when it validates better.

    Args:
        targets: ``(type_id, region_id)`` pairs; defaults to every tracked item
            in every configured region.
    """
    if targets is None:
        targets = [
            (type_id, region_id)
            for region_id in settings.region_ids
            for type_id in TRACKED_TYPE_IDS
        ]

    promoted, kept, failed = 0, 0, 0

    for type_id, region_id in targets:
        try:
            result = await train_item(
                type_id=type_id,
                region_id=region_id,
                promote_only_if_better=True,
            )
        except InsufficientData as e:
            logger.warning(f"Skipping {type_id}: {e}")
            failed += 1
            continue
        except Exception as e:
            logger.exception(f"Retraining failed for {type_id}: {e}")
            failed += 1
            continue

        if result.promoted:
            promoted += 1
            logger.info(
                f"Promoted {result.item_name} {result.model_version} "
                f"(val MAPE {result.metrics['val_mape']:.2f}%)"
            )
        else:
            kept += 1

    logger.info(f"Retraining complete: {promoted} promoted, {kept} kept, {failed} failed")

    if failed:
        await send_alert(
            title="Retraining finished with failures",
            message=f"{failed} item(s) could not be retrained",
            severity="warning",
            context={"promoted": promoted, "kept": kept, "failed": failed},
        )


# --------------------------------------------------------------------------- #
# Wiring
# --------------------------------------------------------------------------- #

def build_scheduler() -> AsyncIOScheduler:
    """Create the scheduler with every job registered."""
    scheduler = AsyncIOScheduler(timezone="UTC")

    defaults = dict(
        max_instances=1,
        coalesce=True,
        misfire_grace_time=MISFIRE_GRACE_SECONDS,
        replace_existing=True,
    )

    scheduler.add_job(
        collect_market_data,
        IntervalTrigger(minutes=settings.collection_interval_minutes),
        id="market_collector",
        name="Order Snapshot Collection",
        **defaults,
    )

    scheduler.add_job(
        compute_features,
        IntervalTrigger(hours=settings.feature_computation_interval_hours),
        id="feature_computer",
        name="Feature Computation",
        **defaults,
    )

    # ESI refreshes market history once a day and caches it for about an hour.
    scheduler.add_job(
        collect_market_history,
        CronTrigger(hour=settings.history_collection_hour, minute=0),
        id="history_collector",
        name="Daily Market History Collection",
        **defaults,
    )

    scheduler.add_job(
        collect_reference_prices,
        CronTrigger(hour=(settings.history_collection_hour + 1) % 24, minute=15),
        id="reference_prices",
        name="Daily Reference Price Collection",
        **defaults,
    )

    # Scoring runs after history lands so realized prices exist to match on.
    scheduler.add_job(
        evaluate_predictions,
        CronTrigger(hour=(settings.history_collection_hour + 1) % 24, minute=30),
        id="prediction_evaluator",
        name="Daily Prediction Scoring",
        **defaults,
    )

    scheduler.add_job(
        generate_all_predictions,
        CronTrigger(hour=settings.prediction_hour, minute=0),
        id="prediction_generator",
        name="Daily Prediction Generation",
        **defaults,
    )

    scheduler.add_job(
        check_drift,
        CronTrigger(hour=settings.drift_check_hour, minute=0),
        id="drift_checker",
        name="Daily Drift Detection",
        **defaults,
    )

    # settings.retraining_day is normalized ("sunday" -> "sun") by the config
    # validator, so CronTrigger can no longer reject it at startup.
    scheduler.add_job(
        retrain_models,
        CronTrigger(
            day_of_week=settings.retraining_day,
            hour=settings.retraining_hour,
        ),
        id="model_retrainer",
        name="Weekly Model Retraining",
        **defaults,
    )

    scheduler.add_listener(_on_job_error, EVENT_JOB_ERROR)

    return scheduler


def _on_job_error(event):
    """Backstop for anything the @job decorator did not catch."""
    logger.error(f"Scheduled job {event.job_id} raised: {event.exception}")


async def main():
    """Run scheduler daemon"""
    logger.info("Starting scheduler daemon...")

    scheduler = build_scheduler()
    scheduler.start()

    for scheduled in scheduler.get_jobs():
        logger.info(f"  {scheduled.name}: next run {scheduled.next_run_time}")

    logger.info("Scheduler started. Press Ctrl+C to exit.")

    try:
        await asyncio.Event().wait()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Shutting down scheduler...")
        scheduler.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
