#!/usr/bin/env python3
"""Train model and generate predictions for a specific item"""

import asyncio
import sys
import logging
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def get_item_name(type_id: int) -> str:
    """Get item name from database"""
    async with get_session() as session:
        result = await session.execute(
            select(Item).where(Item.type_id == type_id)
        )
        item = result.scalar_one_or_none()
        return item.name if item else f"Unknown Item ({type_id})"


async def train_and_predict(
    type_id: int,
    region_id: int = 10000002,
    prediction_days: int = 30,
    epochs: int = 50
):
    """Train model and generate predictions for an item"""

    item_name = await get_item_name(type_id)
    logger.info(f"Training model for: {item_name} (type_id={type_id})")

    # Fetch historical data
    async with get_session() as session:
        result = await session.execute(
            select(MarketHistory)
            .where(MarketHistory.type_id == type_id)
            .where(MarketHistory.region_id == region_id)
            .order_by(MarketHistory.time)
        )
        records = result.scalars().all()

    if not records:
        logger.error(f"No historical data found for type_id={type_id}")
        return None

    logger.info(f"Retrieved {len(records)} historical records")

    # Convert to DataFrame
    df = pd.DataFrame([{
        'time': r.time,
        'average': float(r.average) if r.average else 0.0,
        'volume': r.volume if r.volume else 0,
    } for r in records])

    # Remove any rows with zero price
    df = df[df['average'] > 0]
    logger.info(f"After filtering: {len(df)} records with valid prices")

    if len(df) < settings.lookback_window + 10:
        logger.error(f"Not enough data! Need at least {settings.lookback_window + 10} records")
        return None

    # Show price statistics
    logger.info(f"Price range: {df['average'].min():,.2f} - {df['average'].max():,.2f} ISK")
    logger.info(f"Average price: {df['average'].mean():,.2f} ISK")

    # Prepare sequences for LSTM
    scaler = MinMaxScaler()
    scaled_prices = scaler.fit_transform(df[['average']])

    lookback = settings.lookback_window
    sequences = []
    targets = []

    for i in range(lookback, len(scaled_prices)):
        sequences.append(scaled_prices[i-lookback:i])
        targets.append(scaled_prices[i][0])

    sequences = np.array(sequences)
    targets = np.array(targets)

    logger.info(f"Created {len(sequences)} training sequences")

    # Create dataset
    dataset = MarketDataset(sequences, targets)

    # Train model
    logger.info("Training LSTM model...")
    trainer = ModelTrainer(
        input_size=1,
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
        learning_rate=0.001,
        epochs=epochs,
        batch_size=32,
    )

    metrics = trainer.train(dataset, val_split=0.2, experiment_name=None)

    logger.info(f"Training complete!")
    logger.info(f"Best validation loss: {metrics['best_val_loss']:.6f}")

    # Generate predictions
    logger.info(f"Generating {prediction_days}-day predictions...")

    # Use the last lookback_window prices as input
    last_sequence = scaled_prices[-lookback:].reshape(1, lookback, 1)

    predictions = []
    current_sequence = torch.FloatTensor(last_sequence)

    trainer.model.eval()
    with torch.no_grad():
        for day in range(prediction_days):
            # Predict next value
            pred = trainer.model(current_sequence)
            pred_value = pred.cpu().numpy()[0][0]
            predictions.append(pred_value)

            # Update sequence for next prediction
            current_sequence = torch.cat([
                current_sequence[:, 1:, :],
                torch.FloatTensor([[[pred_value]]])
            ], dim=1)

    # Inverse transform predictions
    predictions_scaled = np.array(predictions).reshape(-1, 1)
    predictions_actual = scaler.inverse_transform(predictions_scaled)

    # Create prediction DataFrame
    last_date = df['time'].max()
    prediction_dates = [last_date + timedelta(days=i+1) for i in range(prediction_days)]

    predictions_df = pd.DataFrame({
        'date': prediction_dates,
        'predicted_price': predictions_actual.flatten(),
    })

    # Add historical context
    historical_recent = df.tail(30)[['time', 'average']].copy()
    historical_recent.columns = ['date', 'actual_price']

    logger.info("\n" + "="*80)
    logger.info(f"PRICE PREDICTIONS FOR {item_name.upper()}")
    logger.info("="*80)
    logger.info(f"\nCurrent Price (latest): {df['average'].iloc[-1]:,.2f} ISK")
    logger.info(f"Predicted Price (30 days): {predictions_actual[-1][0]:,.2f} ISK")
    logger.info(f"Change: {((predictions_actual[-1][0] / df['average'].iloc[-1]) - 1) * 100:+.2f}%")
    logger.info("\n" + "="*80)
    logger.info("\nDay-by-Day Predictions (First 10 days):")
    logger.info("-" * 80)

    for idx, row in predictions_df.head(10).iterrows():
        logger.info(f"Day {idx+1:2d} ({row['date'].strftime('%Y-%m-%d')}): {row['predicted_price']:>15,.2f} ISK")

    logger.info("\n" + "="*80)

    return {
        'item_name': item_name,
        'type_id': type_id,
        'region_id': region_id,
        'current_price': float(df['average'].iloc[-1]),
        'predicted_price_30d': float(predictions_actual[-1][0]),
        'percent_change': float(((predictions_actual[-1][0] / df['average'].iloc[-1]) - 1) * 100),
        'predictions': predictions_df.to_dict('records'),
        'historical': historical_recent.to_dict('records'),
        'metrics': metrics,
    }


async def main():
    """Main prediction function"""
    import os

    # Get type_id from command line or use default
    type_id = int(sys.argv[1]) if len(sys.argv) > 1 else 40520
    prediction_days = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    # Set MLflow URI if not set
    if 'MLFLOW_TRACKING_URI' not in os.environ:
        os.environ['MLFLOW_TRACKING_URI'] = 'http://mlflow:5000'

    result = await train_and_predict(
        type_id=type_id,
        region_id=10000002,  # The Forge (Jita)
        prediction_days=prediction_days,
        epochs=50
    )

    if result:
        logger.info("\nPrediction complete! Results saved above.")


if __name__ == "__main__":
    asyncio.run(main())
