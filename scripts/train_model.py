#!/usr/bin/env python3
"""Train LSTM price prediction models.

Each item gets its own checkpoint under ``MODEL_DIR`` -- see
``src/models/artifacts.py``. Training one item never overwrites another's.

Examples::

    python scripts/train_model.py --type-id 44992                 # PLEX
    python scripts/train_model.py --all                           # every tracked item
    python scripts/train_model.py --all --promote-only-if-better  # what the weekly job does
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.constants import TRACKED_TYPE_IDS, item_name
from src.models.pipeline import InsufficientData, train_item

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--type-id", type=int, help="Item type_id to train")
    target.add_argument("--all", action="store_true", help="Train every tracked item")

    parser.add_argument("--region-id", type=int, default=None,
                        help=f"Region id (default: {settings.region_ids[0]})")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lookback", type=int, default=settings.lookback_window)
    parser.add_argument("--val-split", type=float, default=0.2,
                        help="Fraction of the most recent samples held out")
    parser.add_argument("--promote-only-if-better", action="store_true",
                        help="Keep the existing checkpoint unless the new one validates better")
    return parser.parse_args(argv)


async def main(argv=None):
    args = parse_args(argv)
    region_id = args.region_id or settings.region_ids[0]
    type_ids = TRACKED_TYPE_IDS if args.all else [args.type_id]

    trained, skipped = 0, 0

    for type_id in type_ids:
        logger.info("=" * 70)
        logger.info(f"Training {item_name(type_id)} (type_id={type_id}, region={region_id})")
        try:
            result = await train_item(
                type_id=type_id,
                region_id=region_id,
                epochs=args.epochs,
                lookback=args.lookback,
                val_split=args.val_split,
                promote_only_if_better=args.promote_only_if_better,
            )
        except InsufficientData as e:
            logger.warning(f"Skipped: {e}")
            skipped += 1
            continue

        trained += 1
        logger.info(
            f"{'Promoted' if result.promoted else 'Kept previous'} - "
            f"version {result.model_version or 'n/a'}"
        )
        logger.info(
            f"  val MAE {result.metrics['val_mae']:,.2f} ISK | "
            f"RMSE {result.metrics['val_rmse']:,.2f} | "
            f"MAPE {result.metrics['val_mape']:.2f}% | "
            f"directional {result.metrics['val_directional_accuracy']:.1f}%"
        )
        for baseline, metrics in result.baselines.items():
            logger.info(f"  baseline {baseline:<18} MAPE {metrics['mape']:.2f}%")

    logger.info("=" * 70)
    logger.info(f"Training complete: {trained} trained, {skipped} skipped")
    logger.info(f"Artifacts in {settings.model_dir}")
    return 0 if trained else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
