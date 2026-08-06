#!/usr/bin/env python3
"""Check for data and prediction drift on all tracked items.

Reports two different things:

* **Data drift** -- the input distribution moved (Evidently).
* **Prediction MAPE** -- how wrong recently scored predictions actually were.
  This is empty until ``scripts/evaluate_predictions.py`` has run.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.constants import TRACKED_TYPE_IDS, item_name, region_name
from src.monitoring import DriftDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region-id", type=int, default=None,
                        help=f"Region id (default: {settings.region_ids[0]})")
    parser.add_argument("--reference-days", type=int, default=30)
    parser.add_argument("--current-days", type=int, default=7)
    parser.add_argument("--threshold", type=float, default=settings.drift_threshold,
                        help="Share of drifted features that counts as dataset drift")
    parser.add_argument("--output", default="data/drift_summary.json")
    parser.add_argument("--no-reports", action="store_true",
                        help="Skip writing per-item HTML reports")
    return parser.parse_args(argv)


async def main(argv=None):
    args = parse_args(argv)
    region_id = args.region_id or settings.region_ids[0]

    print("\n" + "=" * 80)
    print("EVE ONLINE - DRIFT DETECTION")
    print(f"{len(TRACKED_TYPE_IDS)} items in {region_name(region_id)}")
    print("=" * 80)

    detector = DriftDetector(
        reference_window_days=args.reference_days,
        current_window_days=args.current_days,
        drift_threshold=args.threshold,
    )

    print(f"\nReference window: {detector.reference_window_days} days")
    print(f"Current window:   {detector.current_window_days} days")
    print(f"Drift threshold:  {detector.drift_threshold}\n")

    items = [(type_id, region_id) for type_id in TRACKED_TYPE_IDS]
    reports = await detector.check_all_items(items, save_reports=not args.no_reports)

    data_drift = target_drift = 0

    print("=" * 80)
    print("RESULTS")
    print("=" * 80)

    for report in reports:
        print(f"\n{item_name(report.type_id)} (type_id={report.type_id})")
        print(f"  Data drift:   {'YES' if report.data_drift_detected else 'no'}")
        print(f"  Target drift: {'YES' if report.prediction_drift_detected else 'no'}")
        print(f"  Drift score:  {report.drift_score:.3f}")

        if report.prediction_mape is not None:
            print(f"  Prediction MAPE (last {args.current_days}d): {report.prediction_mape:.2f}%")
            if report.reference_prediction_mape is not None:
                print(f"  Reference MAPE: {report.reference_prediction_mape:.2f}%")
        else:
            print("  Prediction MAPE: no scored predictions yet")

        if report.drifted_features:
            print(f"  Drifted features: {', '.join(report.drifted_features)}")
        if "error" in report.metrics:
            print(f"  Error: {report.metrics['error']}")
        if report.report_path:
            print(f"  Report: {report.report_path}")

        data_drift += bool(report.data_drift_detected)
        target_drift += bool(report.prediction_drift_detected)

    await detector.save_drift_summary(reports, args.output)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Items checked:            {len(reports)}")
    print(f"Items with data drift:    {data_drift}")
    print(f"Items with target drift:  {target_drift}")
    print(f"Summary written to:       {args.output}")

    if data_drift or target_drift:
        print("\nDrift detected. Suggested actions:")
        print("  1. Review the HTML reports in data/drift_reports/")
        print("  2. Check for market events or patches around the current window")
        print("  3. Retrain: python scripts/train_model.py --all --promote-only-if-better")
        print("  (the scheduler does this automatically when RETRAIN_ON_DRIFT is set)")
    else:
        print("\nNo significant drift detected.")

    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
