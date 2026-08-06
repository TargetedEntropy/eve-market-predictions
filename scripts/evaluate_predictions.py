#!/usr/bin/env python3
"""Score stored predictions against realized market prices.

Fills ``predictions.actual_price`` for elapsed horizons and aggregates the
results into the ``model_performance`` table. Run it after the daily history
collection (the scheduler does this automatically).
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.constants import item_name
from src.monitoring.evaluation import backfill_actual_prices, compute_model_performance

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-age-days", type=int, default=120,
                        help="Ignore predictions older than this")
    return parser.parse_args(argv)


async def main(argv=None):
    args = parse_args(argv)

    scored = await backfill_actual_prices(max_age_days=args.max_age_days)
    print(f"Scored {scored} prediction(s) against realized prices")

    performance = await compute_model_performance(max_age_days=args.max_age_days)
    if not performance:
        print("No scored predictions to aggregate yet.")
        print("Generate some with scripts/predict_all_items.py and wait for the horizon "
              "to elapse.")
        return 0

    print("\n" + "=" * 92)
    print(f"{'Item':<24}{'Horizon':>8}{'N':>6}{'MAE':>14}{'RMSE':>14}{'MAPE':>9}{'Dir%':>9}")
    print("=" * 92)
    for row in sorted(performance, key=lambda r: (r["type_id"], r["horizon_days"])):
        print(
            f"{item_name(row['type_id'])[:23]:<24}"
            f"{row['horizon_days']:>8}"
            f"{row['n_samples']:>6}"
            f"{(row['mae'] or 0):>14,.0f}"
            f"{(row['rmse'] or 0):>14,.0f}"
            f"{(row['mape'] or 0):>8.2f}%"
            f"{(row['directional_accuracy'] or 0):>8.1f}%"
        )
    print("=" * 92)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
