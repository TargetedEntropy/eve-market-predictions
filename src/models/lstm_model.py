"""PyTorch LSTM model for price prediction"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple


class MarketDataset(Dataset):
    """Dataset for time series market data"""

    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        """
        Args:
            sequences: Array of shape (n_samples, lookback_window, n_features)
            targets: Array of shape (n_samples,)
        """
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.targets[idx]


class MarketLSTM(nn.Module):
    """LSTM neural network for price prediction"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1,
    ):
        """
        Args:
            input_size: Number of input features
            hidden_size: Number of LSTM hidden units
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            output_size: Number of output values (default: 1 for price prediction)
        """
        super(MarketLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)

        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # LSTM forward pass
        # lstm_out shape: (batch_size, seq_len, hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Take the last timestep output
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size)

        # Pass through fully connected layers
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out

    def predict(self, x: torch.Tensor) -> np.ndarray:
        """
        Make predictions (inference mode)

        Args:
            x: Input tensor

        Returns:
            NumPy array of predictions
        """
        self.eval()
        with torch.no_grad():
            predictions = self.forward(x)
        return predictions.cpu().numpy()
