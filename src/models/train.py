"""Model training pipeline with MLflow integration"""

import logging
from pathlib import Path
from typing import Tuple, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import mlflow
import mlflow.pytorch
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

from src.config import settings
from src.models.lstm_model import MarketLSTM, MarketDataset

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Train LSTM model with MLflow tracking"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 50,
        patience: int = 10,
        device: Optional[str] = None,
    ):
        """
        Args:
            input_size: Number of input features
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            learning_rate: Learning rate
            batch_size: Batch size
            epochs: Maximum number of epochs
            patience: Early stopping patience
            device: Device to train on (cuda/cpu)
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience

        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Using device: {self.device}")

        # Model
        self.model = None
        self.optimizer = None
        self.criterion = None

    def _initialize_model(self):
        """Initialize model, optimizer, and loss function"""
        self.model = MarketLSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )

        # Using Huber loss (robust to outliers)
        self.criterion = nn.HuberLoss()

        logger.info(
            f"Model initialized: {sum(p.numel() for p in self.model.parameters())} parameters"
        )

    def train(
        self,
        dataset: MarketDataset,
        val_split: float = 0.2,
        experiment_name: Optional[str] = None,
    ) -> dict:
        """
        Train the model

        Args:
            dataset: MarketDataset instance
            val_split: Validation set split ratio
            experiment_name: MLflow experiment name

        Returns:
            Dictionary of training metrics
        """
        # Set up MLflow
        if experiment_name is None:
            experiment_name = settings.mlflow_experiment_name

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name)

        # Start MLflow run
        with mlflow.start_run():
            # Log hyperparameters
            mlflow.log_params(
                {
                    "input_size": self.input_size,
                    "hidden_size": self.hidden_size,
                    "num_layers": self.num_layers,
                    "dropout": self.dropout,
                    "learning_rate": self.learning_rate,
                    "batch_size": self.batch_size,
                    "epochs": self.epochs,
                }
            )

            # Initialize model
            self._initialize_model()

            # Split dataset
            val_size = int(len(dataset) * val_split)
            train_size = len(dataset) - val_size

            train_dataset, val_dataset = random_split(
                dataset, [train_size, val_size]
            )

            # Create data loaders
            train_loader = DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True
            )

            val_loader = DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False
            )

            logger.info(f"Training samples: {train_size}, Validation samples: {val_size}")

            # Training loop
            best_val_loss = float("inf")
            patience_counter = 0
            train_losses = []
            val_losses = []

            for epoch in range(self.epochs):
                # Training phase
                self.model.train()
                train_loss = 0.0

                for sequences, targets in train_loader:
                    sequences = sequences.to(self.device)
                    targets = targets.to(self.device)

                    # Forward pass
                    self.optimizer.zero_grad()
                    outputs = self.model(sequences)
                    loss = self.criterion(outputs.squeeze(), targets)

                    # Backward pass
                    loss.backward()
                    self.optimizer.step()

                    train_loss += loss.item()

                avg_train_loss = train_loss / len(train_loader)
                train_losses.append(avg_train_loss)

                # Validation phase
                self.model.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for sequences, targets in val_loader:
                        sequences = sequences.to(self.device)
                        targets = targets.to(self.device)

                        outputs = self.model(sequences)
                        loss = self.criterion(outputs.squeeze(), targets)
                        val_loss += loss.item()

                avg_val_loss = val_loss / len(val_loader)
                val_losses.append(avg_val_loss)

                # Log metrics
                mlflow.log_metrics(
                    {
                        "train_loss": avg_train_loss,
                        "val_loss": avg_val_loss,
                    },
                    step=epoch,
                )

                logger.info(
                    f"Epoch {epoch+1}/{self.epochs} - "
                    f"Train Loss: {avg_train_loss:.4f}, "
                    f"Val Loss: {avg_val_loss:.4f}"
                )

                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0

                    # Save best model
                    self.save_model(settings.model_path)
                    logger.info(f"New best model saved (val_loss: {best_val_loss:.4f})")

                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        break

            # Log final metrics
            mlflow.log_metric("best_val_loss", best_val_loss)

            # Log model
            mlflow.pytorch.log_model(self.model, "model")

            logger.info("Training complete")

            return {
                "train_losses": train_losses,
                "val_losses": val_losses,
                "best_val_loss": best_val_loss,
            }

    def evaluate(
        self, test_loader: DataLoader
    ) -> Tuple[float, float, float, float]:
        """
        Evaluate model on test set

        Args:
            test_loader: DataLoader for test set

        Returns:
            Tuple of (MAE, RMSE, MAPE, directional_accuracy)
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call train() first.")

        self.model.eval()

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for sequences, targets in test_loader:
                sequences = sequences.to(self.device)
                targets = targets.cpu().numpy()

                outputs = self.model(sequences)
                predictions = outputs.cpu().numpy().squeeze()

                all_predictions.extend(predictions)
                all_targets.extend(targets)

        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)

        # Calculate metrics
        mae = mean_absolute_error(all_targets, all_predictions)
        rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
        mape = np.mean(np.abs((all_targets - all_predictions) / all_targets)) * 100

        # Directional accuracy
        if len(all_targets) > 1:
            direction_true = np.diff(all_targets) > 0
            direction_pred = np.diff(all_predictions) > 0
            directional_accuracy = np.mean(direction_true == direction_pred) * 100
        else:
            directional_accuracy = 0.0

        logger.info(
            f"Test Metrics - MAE: {mae:.4f}, RMSE: {rmse:.4f}, "
            f"MAPE: {mape:.2f}%, Directional Accuracy: {directional_accuracy:.2f}%"
        )

        return mae, rmse, mape, directional_accuracy

    def save_model(self, path: str):
        """Save model to disk"""
        if self.model is None:
            raise RuntimeError("Model not initialized")

        # Create directory if it doesn't exist
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "input_size": self.input_size,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
            },
            path,
        )

        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load model from disk"""
        checkpoint = torch.load(path, map_location=self.device)

        self.input_size = checkpoint["input_size"]
        self.hidden_size = checkpoint["hidden_size"]
        self.num_layers = checkpoint["num_layers"]
        self.dropout = checkpoint["dropout"]

        self._initialize_model()
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.model.eval()

        logger.info(f"Model loaded from {path}")
