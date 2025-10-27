#!/usr/bin/env python3
"""Check model metrics from saved checkpoint"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

model_path = Path("data/models/best_model.pth")

if model_path.exists():
    checkpoint = torch.load(model_path, map_location='cpu')

    print("Model Configuration:")
    print(f"  Input size: {checkpoint.get('input_size')}")
    print(f"  Hidden size: {checkpoint.get('hidden_size')}")
    print(f"  Num layers: {checkpoint.get('num_layers')}")
    print(f"  Dropout: {checkpoint.get('dropout')}")
    print()

    print("Model keys:", list(checkpoint.keys()))
else:
    print(f"Model not found at {model_path}")
