#!/usr/bin/env python3
"""Generate price predictions for one item from its saved model.

This predicts from the checkpoint on disk; it does **not** retrain unless you
pass ``--train``. (The previous version retrained on every run and overwrote a
single shared ``best_model.pth``, so "predicting" one item silently replaced
the model the API was serving for every other item.)

The forecast is autoregressive: the model predicts one day ahead, that value is
fed back in, and so on. Error compounds with horizon -- day 30 is a trend
indication, not a quote.

Every run writes rows to the ``predictions`` table so accuracy can be scored
later by ``scripts/evaluate_predictions.py``.

Examples::

    python scripts/predict_item.py 40520              # Large Skill Injector, 30 days
    python scripts/predict_item.py 44992 --days 7
    python scripts/predict_item.py 44992 --train      # retrain first
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.models.pipeline import (
    InsufficientData,
    ModelNotTrained,
    generate_predictions,
    train_item,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("type_id", type=int, help="Item type_id to forecast")
    parser.add_argument("--days", type=int, default=settings.prediction_horizon_days,
                        help="Forecast horizon in days")
    parser.add_argument("--region-id", type=int, default=None,
                        help=f"Region id (default: {settings.region_ids[0]})")
    parser.add_argument("--train", action="store_true",
                        help="Retrain this item's model before forecasting")
    parser.add_argument("--epochs", type=int, default=50, help="Epochs when --train is used")
    parser.add_argument("--json", dest="json_path", default=None,
                        help="Also write the result to this JSON file")
    parser.add_argument("--no-store", action="store_true",
                        help="Skip writing rows to the predictions table")
    return parser.parse_args(argv)


def print_report(result):
    print("\n" + "=" * 80)
    print(f"{result['item_name'].upper()} - {result['horizon_days']}-DAY PRICE FORECAST")
    print("=" * 80)
    print(f"Model version:       {result['model_version']}")
    confidence = result["confidence"]
    print(f"Confidence:          "
          f"{'n/a' if confidence is None else f'{confidence:.2%} (1 - validation MAPE)'}")
    print(f"Last observed day:   {result['last_observed_date']:%Y-%m-%d}")
    print(f"Current price:       {result['current_price']:>20,.0f} ISK")
    print(f"Predicted price:     {result['predicted_price']:>20,.0f} ISK")
    print(f"Expected change:     {result['percent_change']:>19.2f}%")
    print("-" * 80)
    for row in result["predictions"][:10]:
        print(f"Day {row['day']:2d} ({row['date']}): {row['predicted_price']:>20,.0f} ISK")
    if len(result["predictions"]) > 10:
        print("...")
        for row in result["predictions"][-3:]:
            print(f"Day {row['day']:2d} ({row['date']}): {row['predicted_price']:>20,.0f} ISK")
    print("=" * 80)


async def main(argv=None):
    args = parse_args(argv)
    region_id = args.region_id or settings.region_ids[0]

    if args.train:
        try:
            training = await train_item(
                type_id=args.type_id, region_id=region_id, epochs=args.epochs
            )
        except InsufficientData as e:
            logger.error(str(e))
            return 1
        logger.info(
            f"Trained {training.item_name}: val MAPE {training.metrics['val_mape']:.2f}%"
        )

    try:
        result = await generate_predictions(
            type_id=args.type_id,
            region_id=region_id,
            horizon_days=args.days,
            store=not args.no_store,
        )
    except (ModelNotTrained, InsufficientData) as e:
        logger.error(str(e))
        return 1

    print_report(result)

    if args.json_path:
        output = Path(args.json_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, default=str))
        print(f"\nSaved to {output}")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
