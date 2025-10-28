#!/usr/bin/env python3
"""Train models and save predictions for all tracked items"""

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


# Items to predict
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


async def predict_item(type_id: int, item_name: str):
    """Train and predict for a specific item"""

    region_id = 10000002  # The Forge

    print(f"\n{'='*80}")
    print(f"Processing: {item_name} (type_id={type_id})")
    print(f"{'='*80}")

    try:
        # Fetch historical data
        async with get_session() as session:
            result = await session.execute(
                select(MarketHistory)
                .where(MarketHistory.type_id == type_id)
                .where(MarketHistory.region_id == region_id)
                .order_by(MarketHistory.time)
            )
            records = result.scalars().all()

        if len(records) < 50:
            print(f"⚠ Insufficient data: {len(records)} records (need at least 50)")
            return None

        print(f"✓ Found {len(records)} historical records")

        # Convert to DataFrame
        df = pd.DataFrame([{
            'time': r.time,
            'average': float(r.average) if r.average else 0.0,
        } for r in records])

        df = df[df['average'] > 0]

        if len(df) < 50:
            print(f"⚠ Insufficient valid data: {len(df)} records after filtering")
            return None

        print(f"  Price range: {df['average'].min():,.0f} - {df['average'].max():,.0f} ISK")
        print(f"  Average price: {df['average'].mean():,.0f} ISK")

        # Prepare data
        scaler = MinMaxScaler()
        scaled_prices = scaler.fit_transform(df[['average']])

        lookback = 30
        sequences = []
        targets = []

        for i in range(lookback, len(scaled_prices)):
            sequences.append(scaled_prices[i-lookback:i])
            targets.append(scaled_prices[i][0])

        if len(sequences) < 20:
            print(f"⚠ Insufficient sequences: {len(sequences)} (need at least 20)")
            return None

        sequences = np.array(sequences)
        targets = np.array(targets)

        dataset = MarketDataset(sequences, targets)

        # Train
        print(f"  Training LSTM model on {len(sequences)} sequences...")
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
        print(f"  ✓ Training complete! Val loss: {metrics['best_val_loss']:.6f}")

        # Generate predictions
        print(f"  Generating 30-day predictions...")

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

        current_price = float(df['average'].iloc[-1])
        predicted_price_30d = float(predictions_actual[-1][0])
        percent_change = float(((predicted_price_30d / current_price) - 1) * 100)

        results = {
            'item_name': item_name,
            'type_id': type_id,
            'region_id': region_id,
            'training_date': datetime.now().isoformat(),
            'current_price': current_price,
            'predicted_price_30d': predicted_price_30d,
            'percent_change': percent_change,
            'best_val_loss': float(metrics['best_val_loss']),
            'data_points': len(records),
            'predictions': []
        }

        for i in range(30):
            results['predictions'].append({
                'day': i + 1,
                'date': prediction_dates[i].strftime('%Y-%m-%d'),
                'predicted_price': float(predictions_actual[i][0])
            })

        # Save to file
        safe_name = item_name.lower().replace(' ', '_')
        output_file = Path(f'data/predictions_{safe_name}.json')
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"  ✓ Current: {current_price:,.0f} ISK")
        print(f"  ✓ Predicted (30d): {predicted_price_30d:,.0f} ISK")
        print(f"  ✓ Change: {percent_change:+.2f}%")
        print(f"  ✓ Saved to: {output_file}")

        return results

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


async def main():
    """Process all items"""

    print("\n" + "="*80)
    print("EVE ONLINE - BULK PRICE PREDICTION")
    print(f"Processing {len(ITEMS)} items...")
    print("="*80)

    results = []
    successful = 0
    failed = 0

    for item in ITEMS:
        result = await predict_item(item['type_id'], item['name'])
        if result:
            results.append(result)
            successful += 1
        else:
            failed += 1

    # Save summary
    summary = {
        'generated_at': datetime.now().isoformat(),
        'total_items': len(ITEMS),
        'successful': successful,
        'failed': failed,
        'items': results
    }

    summary_file = Path('data/predictions_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total items processed: {len(ITEMS)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"\nSummary saved to: {summary_file}")

    if results:
        print("\nTop 5 Expected Gainers:")
        sorted_by_gain = sorted(results, key=lambda x: x['percent_change'], reverse=True)[:5]
        for i, item in enumerate(sorted_by_gain, 1):
            print(f"{i}. {item['item_name']}: {item['percent_change']:+.2f}%")

        print("\nTop 5 Expected Losers:")
        sorted_by_loss = sorted(results, key=lambda x: x['percent_change'])[:5]
        for i, item in enumerate(sorted_by_loss, 1):
            print(f"{i}. {item['item_name']}: {item['percent_change']:+.2f}%")

    print("\n" + "="*80)


if __name__ == "__main__":
    asyncio.run(main())
