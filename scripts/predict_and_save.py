#!/usr/bin/env python3
"""Train model and save predictions for Large Skill Injector"""

import asyncio
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from sqlalchemy import select

from src.config import settings
from src.database import get_session, MarketHistory, Item
from src.models import MarketLSTM, MarketDataset, ModelTrainer


async def main():
    """Train and predict for Large Skill Injector"""

    type_id = 40520  # Large Skill Injector
    region_id = 10000002  # The Forge

    print(f"Training model for type_id={type_id}...")

    # Fetch historical data
    async with get_session() as session:
        result = await session.execute(
            select(MarketHistory)
            .where(MarketHistory.type_id == type_id)
            .where(MarketHistory.region_id == region_id)
            .order_by(MarketHistory.time)
        )
        records = result.scalars().all()

        # Get item name
        item_result = await session.execute(
            select(Item).where(Item.type_id == type_id)
        )
        item = item_result.scalar_one_or_none()
        item_name = item.name if item else "Large Skill Injector"

    print(f"Found {len(records)} historical records for {item_name}")

    # Convert to DataFrame
    df = pd.DataFrame([{
        'time': r.time,
        'average': float(r.average) if r.average else 0.0,
    } for r in records])

    df = df[df['average'] > 0]

    print(f"Price range: {df['average'].min():,.0f} - {df['average'].max():,.0f} ISK")
    print(f"Average price: {df['average'].mean():,.0f} ISK")

    # Prepare data
    scaler = MinMaxScaler()
    scaled_prices = scaler.fit_transform(df[['average']])

    lookback = 30
    sequences = []
    targets = []

    for i in range(lookback, len(scaled_prices)):
        sequences.append(scaled_prices[i-lookback:i])
        targets.append(scaled_prices[i][0])

    sequences = np.array(sequences)
    targets = np.array(targets)

    dataset = MarketDataset(sequences, targets)

    # Train
    print(f"\nTraining LSTM model on {len(sequences)} sequences...")
    trainer = ModelTrainer(
        input_size=1,
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
        learning_rate=0.001,
        epochs=50,
        batch_size=32,
    )

    metrics = trainer.train(dataset, val_split=0.2, experiment_name=None)
    print(f"Training complete! Best val loss: {metrics['best_val_loss']:.6f}")

    # Generate predictions
    print(f"\nGenerating 30-day predictions...")

    last_sequence = scaled_prices[-lookback:].reshape(1, lookback, 1)
    predictions = []
    current_sequence = torch.FloatTensor(last_sequence)

    trainer.model.eval()
    with torch.no_grad():
        for day in range(30):
            pred = trainer.model(current_sequence)
            pred_value = pred.cpu().numpy()[0][0]
            predictions.append(pred_value)

            current_sequence = torch.cat([
                current_sequence[:, 1:, :],
                torch.FloatTensor([[[pred_value]]])
            ], dim=1)

    # Inverse transform
    predictions_scaled = np.array(predictions).reshape(-1, 1)
    predictions_actual = scaler.inverse_transform(predictions_scaled)

    # Create results
    last_date = df['time'].max()
    prediction_dates = [last_date + timedelta(days=i+1) for i in range(30)]

    results = {
        'item_name': item_name,
        'type_id': type_id,
        'region_id': region_id,
        'training_date': datetime.now().isoformat(),
        'current_price': float(df['average'].iloc[-1]),
        'predicted_price_30d': float(predictions_actual[-1][0]),
        'percent_change': float(((predictions_actual[-1][0] / df['average'].iloc[-1]) - 1) * 100),
        'best_val_loss': float(metrics['best_val_loss']),
        'predictions': []
    }

    for i in range(30):
        results['predictions'].append({
            'day': i + 1,
            'date': prediction_dates[i].strftime('%Y-%m-%d'),
            'predicted_price': float(predictions_actual[i][0])
        })

    # Save to file
    output_file = Path('data/predictions_large_skill_injector.json')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n" + "="*80)
    print(f"LARGE SKILL INJECTOR - 30-DAY PRICE PREDICTION")
    print("="*80)
    print(f"\nCurrent Price (Oct 26, 2025): {results['current_price']:>20,.0f} ISK")
    print(f"Predicted Price (Nov 25, 2025): {results['predicted_price_30d']:>20,.0f} ISK")
    print(f"Expected Change: {results['percent_change']:>20.2f}%")
    print(f"\nModel Validation Loss: {results['best_val_loss']:.6f}")
    print("\n" + "="*80)
    print("\nDay-by-Day Predictions:")
    print("-" * 80)

    for pred in results['predictions'][:10]:
        print(f"Day {pred['day']:2d} ({pred['date']}): {pred['predicted_price']:>20,.0f} ISK")

    print("...")
    for pred in results['predictions'][-3:]:
        print(f"Day {pred['day']:2d} ({pred['date']}): {pred['predicted_price']:>20,.0f} ISK")

    print("\n" + "="*80)
    print(f"\nFull predictions saved to: {output_file}")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())
