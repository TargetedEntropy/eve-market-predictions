"""Model training pipeline with MLflow integration"""

import copy
import logging
import os
import socket
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, Subset

from src.config import settings
from src.models.lstm_model import MarketDataset, MarketLSTM

logger = logging.getLogger(__name__)

# Guards MAPE against zero targets
_EPSILON = 1e-9

#: Cached result of the MLflow reachability probe (None = not probed yet)
_MLFLOW_REACHABLE: Optional[bool] = None


def _mlflow_reachable(uri: str, timeout: float = 2.0) -> bool:
    """Cheap reachability probe for an MLflow tracking server.

    MLflow's own client retries with backoff for minutes against a dead
    server, which would stall the unattended retraining job. A 2s TCP connect
    answers the question first. Local (file/sqlite) stores always pass.
    """
    global _MLFLOW_REACHABLE
    if _MLFLOW_REACHABLE is not None:
        return _MLFLOW_REACHABLE

    parsed = urlparse(uri)
    if parsed.scheme not in ("http", "https"):
        _MLFLOW_REACHABLE = True
        return True

    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((parsed.hostname, port), timeout=timeout):
            _MLFLOW_REACHABLE = True
    except OSError as e:
        logger.warning(f"MLflow tracking server {uri} unreachable ({e}); skipping tracking")
        _MLFLOW_REACHABLE = False
    return _MLFLOW_REACHABLE


def reset_mlflow_probe() -> None:
    """Forget the cached reachability result (tests, long-lived daemons)."""
    global _MLFLOW_REACHABLE
    _MLFLOW_REACHABLE = None


@contextmanager
def _mlflow_run(experiment_name: str, run_name: Optional[str] = None):
    """Start an MLflow run, degrading to a no-op if the server is unreachable.

    Training must not hang or fail because the tracking server is down -- the
    scheduled retraining job runs unattended.
    """
    if not _mlflow_reachable(settings.mlflow_tracking_uri):
        yield None
        return

    # Bound the client's own retry behaviour as well.
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "2")

    try:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name)
        run = mlflow.start_run(run_name=run_name)
    except Exception as e:
        logger.warning(f"MLflow unavailable ({e}); training without tracking")
        yield None
        return

    try:
        yield run
    finally:
        try:
            mlflow.end_run()
        except Exception:  # pragma: no cover - best effort
            pass


def _safe_mlflow(enabled: bool, call, *args, **kwargs) -> None:
    """Invoke an MLflow logging call, ignoring failures (and skipping it when
    no run is active -- otherwise MLflow silently creates a local ./mlruns
    store on a machine that has no tracking server)."""
    if not enabled:
        return
    try:
        call(*args, **kwargs)
    except Exception as e:
        logger.debug(f"MLflow call failed: {e}")


def chronological_split(dataset: MarketDataset, val_split: float) -> Tuple[Subset, Subset]:
    """Split a time-series dataset by time, not at random.

    ``random_split`` over overlapping lookback windows puts nearly identical
    sequences in both sides and leaks future data into training, which made
    reported validation losses meaningless.
    """
    total = len(dataset)
    val_size = int(total * val_split)
    train_size = total - val_size
    if train_size <= 0:
        raise ValueError("Not enough samples to build a training split")
    return Subset(dataset, range(train_size)), Subset(dataset, range(train_size, total))


def regression_metrics(targets: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
    """MAE / RMSE / MAPE / directional accuracy, safe on degenerate inputs."""
    targets = np.asarray(targets, dtype=float).ravel()
    predictions = np.asarray(predictions, dtype=float).ravel()

    if targets.size == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "mape": float("nan"),
                "directional_accuracy": float("nan")}

    mae = float(mean_absolute_error(targets, predictions))
    rmse = float(np.sqrt(mean_squared_error(targets, predictions)))

    # Ignore zero targets rather than dividing by them (which yields inf).
    nonzero = np.abs(targets) > _EPSILON
    if nonzero.any():
        mape = float(
            np.mean(np.abs((targets[nonzero] - predictions[nonzero]) / targets[nonzero])) * 100
        )
    else:
        mape = float("nan")

    if targets.size > 1:
        direction_true = np.diff(targets) > 0
        direction_pred = np.diff(predictions) > 0
        directional_accuracy = float(np.mean(direction_true == direction_pred) * 100)
    else:
        directional_accuracy = float("nan")

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "directional_accuracy": directional_accuracy,
    }


class ModelTrainer:
    """Train LSTM model with MLflow tracking.

    The best epoch's weights are kept in memory and restored at the end of
    training; persisting them is the caller's job (see
    :mod:`src.models.artifacts`), so training one item can no longer clobber
    another item's checkpoint.
    """

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
        self.best_state = None

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
        run_name: Optional[str] = None,
        extra_params: Optional[Dict[str, object]] = None,
        registered_model_name: Optional[str] = None,
    ) -> dict:
        """
        Train the model on a chronological split of ``dataset``.

        Args:
            dataset: MarketDataset instance, ordered oldest to newest
            val_split: Fraction of the *most recent* samples held out
            experiment_name: MLflow experiment name
            run_name: MLflow run name
            extra_params: Additional params to log (e.g. type_id)

        Returns:
            Dictionary of training metrics
        """
        if experiment_name is None:
            experiment_name = settings.mlflow_experiment_name

        train_dataset, val_dataset = chronological_split(dataset, val_split)

        with _mlflow_run(experiment_name, run_name) as active_run:
            tracking = active_run is not None
            params = {
                "input_size": self.input_size,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "epochs": self.epochs,
                "val_split": val_split,
                "split_strategy": "chronological",
            }
            params.update(extra_params or {})
            _safe_mlflow(tracking, mlflow.log_params, params)

            self._initialize_model()

            train_loader = DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True
            )
            val_loader = DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False
            )

            logger.info(
                f"Training samples: {len(train_dataset)}, "
                f"Validation samples: {len(val_dataset)}"
            )

            best_val_loss = float("inf")
            patience_counter = 0
            train_losses = []
            val_losses = []
            self.best_state = copy.deepcopy(self.model.state_dict())

            for epoch in range(self.epochs):
                self.model.train()
                train_loss = 0.0

                for sequences, targets in train_loader:
                    sequences = sequences.to(self.device)
                    targets = targets.to(self.device)

                    self.optimizer.zero_grad()
                    outputs = self.model(sequences)
                    # squeeze(-1), not squeeze(): a final batch of size 1 would
                    # otherwise collapse to a scalar and broadcast silently.
                    loss = self.criterion(outputs.squeeze(-1), targets)

                    loss.backward()
                    self.optimizer.step()

                    train_loss += loss.item()

                avg_train_loss = train_loss / max(len(train_loader), 1)
                train_losses.append(avg_train_loss)

                self.model.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for sequences, targets in val_loader:
                        sequences = sequences.to(self.device)
                        targets = targets.to(self.device)

                        outputs = self.model(sequences)
                        loss = self.criterion(outputs.squeeze(-1), targets)
                        val_loss += loss.item()

                avg_val_loss = val_loss / max(len(val_loader), 1)
                val_losses.append(avg_val_loss)

                _safe_mlflow(
                    tracking,
                    mlflow.log_metrics,
                    {"train_loss": avg_train_loss, "val_loss": avg_val_loss},
                    step=epoch,
                )

                logger.info(
                    f"Epoch {epoch+1}/{self.epochs} - "
                    f"Train Loss: {avg_train_loss:.6f}, "
                    f"Val Loss: {avg_val_loss:.6f}"
                )

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    self.best_state = copy.deepcopy(self.model.state_dict())
                    logger.debug(f"New best epoch (val_loss: {best_val_loss:.6f})")
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        break

            # Restore the best epoch before anything is saved or evaluated.
            self.model.load_state_dict(self.best_state)

            _safe_mlflow(tracking, mlflow.log_metric, "best_val_loss", best_val_loss)

            # Register the model so versions/stages exist rather than only
            # anonymous artifacts (no-op when MLflow is unreachable).
            _safe_mlflow(
                tracking,
                mlflow.pytorch.log_model,
                self.model,
                "model",
                registered_model_name=registered_model_name,
            )

            logger.info("Training complete")

            return {
                "train_losses": train_losses,
                "val_losses": val_losses,
                "best_val_loss": best_val_loss,
                "n_train": len(train_dataset),
                "n_val": len(val_dataset),
                "val_dataset": val_dataset,
            }

    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model on a data loader.

        Returns:
            Dict with mae, rmse, mape and directional_accuracy (in the units
            the loader yields -- scaled space unless the caller inverts it).
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call train() first.")

        self.model.eval()

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for sequences, targets in test_loader:
                sequences = sequences.to(self.device)

                outputs = self.model(sequences).squeeze(-1)

                all_predictions.append(outputs.cpu().numpy().ravel())
                all_targets.append(targets.cpu().numpy().ravel())

        predictions = np.concatenate(all_predictions) if all_predictions else np.array([])
        targets = np.concatenate(all_targets) if all_targets else np.array([])

        metrics = regression_metrics(targets, predictions)

        logger.info(
            f"Metrics - MAE: {metrics['mae']:.6f}, RMSE: {metrics['rmse']:.6f}, "
            f"MAPE: {metrics['mape']:.2f}%, "
            f"Directional Accuracy: {metrics['directional_accuracy']:.2f}%"
        )

        return metrics

    def predict_loader(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(targets, predictions)`` arrays for a loader."""
        self.model.eval()
        preds, targs = [], []
        with torch.no_grad():
            for sequences, targets in loader:
                outputs = self.model(sequences.to(self.device)).squeeze(-1)
                preds.append(outputs.cpu().numpy().ravel())
                targs.append(targets.cpu().numpy().ravel())
        if not preds:
            return np.array([]), np.array([])
        return np.concatenate(targs), np.concatenate(preds)

    def save_model(self, path: str):
        """Save the current weights to an explicit path (see also artifacts.py)"""
        if self.model is None:
            raise RuntimeError("Model not initialized")

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.cpu() for k, v in self.model.state_dict().items()}, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load weights from disk (never unpickles arbitrary objects)"""
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        self._initialize_model()
        self.model.load_state_dict(state_dict)
        self.model.eval()
        logger.info(f"Model loaded from {path}")
