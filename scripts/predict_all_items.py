#!/usr/bin/env python3
"""Generate predictions for every tracked item.

Predicts from the saved per-item checkpoints (``--train`` retrains first).
Results go to the ``predictions`` table; the JSON files are an optional export
for offline inspection.

Examples::

    python scripts/predict_all_items.py                 # forecast from saved models
    python scripts/predict_all_items.py --train         # retrain everything first
    python scripts/predict_all_items.py --json-dir data # also write JSON
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.constants import TRACKED_TYPE_IDS, item_name
from src.models.pipeline import (
    InsufficientData,
    ModelNotTrained,
    generate_predictions,
    train_item,
)
from src.utils.timeutils import utcnow

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--region-id", type=int, default=None,
                        help=f"Region id (default: {settings.region_ids[0]})")
    parser.add_argument("--days", type=int, default=settings.prediction_horizon_days)
    parser.add_argument("--train", action="store_true",
                        help="Retrain each item's model before forecasting")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--json-dir", default=None,
                        help="Directory for optional per-item JSON exports")
    parser.add_argument("--no-store", action="store_true",
                        help="Skip writing rows to the predictions table")
    return parser.parse_args(argv)


async def main(argv=None):
    args = parse_args(argv)
    region_id = args.region_id or settings.region_ids[0]

    print("\n" + "=" * 80)
    print("EVE ONLINE - BULK PRICE PREDICTION")
    print(f"Processing {len(TRACKED_TYPE_IDS)} items in region {region_id}")
    print("=" * 80)

    results, failures = [], []

    for type_id in TRACKED_TYPE_IDS:
        name = item_name(type_id)
        print(f"\n--- {name} (type_id={type_id}) ---")

        if args.train:
            try:
                training = await train_item(
                    type_id=type_id, region_id=region_id, epochs=args.epochs
                )
                print(f"  trained: val MAPE {training.metrics['val_mape']:.2f}%")
            except InsufficientData as e:
                print(f"  skipped: {e}")
                failures.append({"type_id": type_id, "item_name": name, "error": str(e)})
                continue
            except Exception as e:
                logger.exception(f"Training failed for {type_id}")
                failures.append({"type_id": type_id, "item_name": name, "error": str(e)})
                continue

        try:
            result = await generate_predictions(
                type_id=type_id,
                region_id=region_id,
                horizon_days=args.days,
                store=not args.no_store,
            )
        except (ModelNotTrained, InsufficientData) as e:
            print(f"  skipped: {e}")
            failures.append({"type_id": type_id, "item_name": name, "error": str(e)})
            continue
        except Exception as e:
            logger.exception(f"Prediction failed for {type_id}")
            failures.append({"type_id": type_id, "item_name": name, "error": str(e)})
            continue

        results.append(result)
        print(f"  current:   {result['current_price']:>18,.0f} ISK")
        print(f"  predicted: {result['predicted_price']:>18,.0f} ISK "
              f"({result['percent_change']:+.2f}% over {result['horizon_days']}d)")

        if args.json_dir:
            out = Path(args.json_dir) / f"predictions_{name.lower().replace(' ', '_')}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(result, indent=2, default=str))

    if args.json_dir:
        summary = {
            "generated_at": utcnow().isoformat(),
            "region_id": region_id,
            "total_items": len(TRACKED_TYPE_IDS),
            "successful": len(results),
            "failed": len(failures),
            "items": results,
            "failures": failures,
        }
        summary_file = Path(args.json_dir) / "predictions_summary.json"
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        summary_file.write_text(json.dumps(summary, indent=2, default=str))
        print(f"\nSummary written to {summary_file}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Successful: {len(results)}   Failed: {len(failures)}")

    if results:
        ranked = sorted(results, key=lambda r: r["percent_change"], reverse=True)
        print("\nTop expected gainers:")
        for i, row in enumerate(ranked[:5], 1):
            print(f"{i}. {row['item_name']}: {row['percent_change']:+.2f}%")
        print("\nTop expected losers:")
        for i, row in enumerate(sorted(results, key=lambda r: r["percent_change"])[:5], 1):
            print(f"{i}. {row['item_name']}: {row['percent_change']:+.2f}%")

    print("=" * 80)
    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
