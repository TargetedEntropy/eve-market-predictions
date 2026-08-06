#!/usr/bin/env python3
"""Show the metadata and validation metrics of every trained model.

Reads the JSON metadata written alongside each checkpoint, so nothing is
unpickled just to inspect a model.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.constants import item_name, region_name
from src.models.artifacts import available_models, checkpoint_path, load_metadata


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--type-id", type=int, default=None,
                        help="Only show this item")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    models = available_models()

    if args.type_id is not None:
        models = [pair for pair in models if pair[0] == args.type_id]

    if not models:
        print(f"No trained models found in {settings.model_dir}")
        print("Train one with: python scripts/train_model.py --all")
        return 1

    for type_id, region_id in models:
        metadata = load_metadata(type_id, region_id)
        if metadata is None:
            continue

        print("=" * 78)
        print(f"{item_name(type_id)} (type_id={type_id}) - {region_name(region_id)}")
        print("=" * 78)
        print(f"  Version:          {metadata.model_version}")
        print(f"  Trained at:       {metadata.trained_at}")
        print(f"  Checkpoint:       {checkpoint_path(type_id, region_id)}")
        print(f"  Architecture:     input={metadata.input_size} hidden={metadata.hidden_size} "
              f"layers={metadata.num_layers} dropout={metadata.dropout}")
        print(f"  Lookback window:  {metadata.lookback_window} days")
        print(f"  Feature columns:  {', '.join(metadata.feature_columns)}")
        print(f"  Train/val split:  {metadata.n_train_samples}/{metadata.n_val_samples} samples "
              f"(chronological, train ends {metadata.train_end})")
        print(f"  Scaler range:     {metadata.scaler.data_min} .. {metadata.scaler.data_max}")

        if metadata.metrics:
            print("  Validation metrics:")
            for key, value in metadata.metrics.items():
                print(f"    {key:<28} {value:,.4f}")

        if metadata.baselines:
            print("  Baselines (validation MAPE):")
            for name, metrics in metadata.baselines.items():
                print(f"    {name:<28} {metrics.get('mape', float('nan')):.2f}%")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
