#!/usr/bin/env python3
"""Check for model drift on all tracked items"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.monitoring import DriftDetector
from src.config import settings


# Items to check for drift
ITEMS = [
    {'name': 'PLEX', 'type_id': 44992},
    {'name': 'Large Skill Injector', 'type_id': 40520},
    {'name': 'Skill Extractor', 'type_id': 40519},
    {'name': 'Tritanium', 'type_id': 34},
    {'name': 'Pyerite', 'type_id': 35},
    {'name': 'Mexallon', 'type_id': 36},
    {'name': 'Isogen', 'type_id': 37},
    {'name': 'Nocxium', 'type_id': 38},
    {'name': 'Zydrine', 'type_id': 39},
    {'name': 'Megacyte', 'type_id': 40},
    {'name': 'Morphite', 'type_id': 11399},
]


async def main():
    """Run drift detection on all items"""

    print("\n" + "="*80)
    print("EVE ONLINE - MODEL DRIFT DETECTION")
    print(f"Checking {len(ITEMS)} items for drift...")
    print("="*80)

    # Initialize drift detector
    detector = DriftDetector(
        reference_window_days=30,  # 30 days of baseline data
        current_window_days=7,     # Compare to last 7 days
        drift_threshold=0.5,       # 50% of features drifted = alert
    )

    # Check drift for all items
    region_id = 10000002  # The Forge (Jita)
    items_to_check = [(item['type_id'], region_id) for item in ITEMS]

    print(f"\nReference window: {detector.reference_window_days} days")
    print(f"Current window: {detector.current_window_days} days")
    print(f"Drift threshold: {detector.drift_threshold}\n")

    # Run drift detection
    reports = await detector.check_all_items(items_to_check, save_reports=True)

    # Print results
    print("\n" + "="*80)
    print("DRIFT DETECTION RESULTS")
    print("="*80)

    items_with_data_drift = 0
    items_with_target_drift = 0

    for item, report in zip(ITEMS, reports):
        print(f"\n{item['name']} (type_id={item['type_id']})")
        print(f"  Data drift detected: {'YES ⚠️' if report.data_drift_detected else 'No'}")
        print(f"  Target drift detected: {'YES ⚠️' if report.prediction_drift_detected else 'No'}")
        print(f"  Drift score: {report.drift_score:.3f}")

        if report.drifted_features:
            print(f"  Drifted features: {', '.join(report.drifted_features)}")

        if report.data_drift_detected:
            items_with_data_drift += 1
        if report.prediction_drift_detected:
            items_with_target_drift += 1

        if 'error' in report.metrics:
            print(f"  Error: {report.metrics['error']}")

        if report.report_path:
            print(f"  Report: {report.report_path}")

    # Save summary
    summary_path = 'data/drift_summary.json'
    await detector.save_drift_summary(reports, summary_path)

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total items checked: {len(reports)}")
    print(f"Items with data drift: {items_with_data_drift}")
    print(f"Items with target drift: {items_with_target_drift}")
    print(f"\nSummary saved to: {summary_path}")

    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    if items_with_data_drift > 0 or items_with_target_drift > 0:
        print("⚠️  Drift detected! Consider the following actions:")
        print("  1. Review drift reports in data/drift_reports/")
        print("  2. Investigate feature changes (market events, patches, etc.)")
        print("  3. Retrain models with recent data")
        print("  4. Update feature engineering pipeline if needed")
        print("  5. Monitor prediction accuracy closely")
    else:
        print("✓ No significant drift detected.")
        print("  Models appear to be performing well on current data.")
        print("  Continue regular monitoring.")

    print("\n" + "="*80)


if __name__ == "__main__":
    asyncio.run(main())
