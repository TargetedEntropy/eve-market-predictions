#!/usr/bin/env python3
"""Train LSTM price prediction model"""

import asyncio
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from sqlalchemy import select

from src.config import settings
from src.database import get_session, MarketHistory
from src.models import MarketLSTM, MarketDataset, ModelTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Train model on PLEX price data"""
    logger.info("Starting model training...")

    # Fetch training data for PLEX
    type_id = 44992  # PLEX
    region_id = 10000002  # The Forge (Jita)

    logger.info(f"Fetching historical data for type_id={type_id}, region_id={region_id}")

    async with get_session() as session:
        result = await session.execute(
            select(MarketHistory)
            .where(MarketHistory.type_id == type_id)
            .where(MarketHistory.region_id == region_id)
            .order_by(MarketHistory.time)
        )
        records = result.scalars().all()

    if not records:
        logger.error("No historical data found!")
        return

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
        return

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
    logger.info("Initializing model trainer...")
    trainer = ModelTrainer(
        input_size=1,
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
        learning_rate=0.001,
        epochs=50,
        batch_size=32,
    )

    logger.info("Starting training...")
    metrics = trainer.train(dataset, val_split=0.2)

    logger.info(f"Training complete!")
    logger.info(f"Best validation loss: {metrics['best_val_loss']:.6f}")
    logger.info(f"Final train loss: {metrics['train_losses'][-1]:.6f}")
    logger.info(f"Final validation loss: {metrics['val_losses'][-1]:.6f}")

    # Note: Model is already saved by trainer.train() as best_model.pth
    logger.info(f"Best model saved at {settings.model_path}")
    logger.info("Training pipeline complete!")


if __name__ == "__main__":
    asyncio.run(main())
