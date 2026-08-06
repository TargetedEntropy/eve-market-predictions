"""Simple forecasting baselines.

An LSTM is only worth its complexity if it beats "tomorrow looks like today".
These run on the same chronological validation window as the model and are
logged alongside it, so a regression is visible instead of implied.
"""

from typing import Dict

import numpy as np


def naive_forecast(window: np.ndarray) -> float:
    """Last observed value."""
    return float(window[-1])


def moving_average_forecast(window: np.ndarray, periods: int = 7) -> float:
    """Mean of the last ``periods`` observations."""
    return float(np.mean(window[-periods:]))


def linear_trend_forecast(window: np.ndarray) -> float:
    """One-step extrapolation of a least-squares line through the window."""
    x = np.arange(len(window), dtype=float)
    slope, intercept = np.polyfit(x, window.astype(float), 1)
    return float(slope * len(window) + intercept)


BASELINES = {
    "naive": naive_forecast,
    "moving_average_7": lambda w: moving_average_forecast(w, 7),
    "linear_trend": linear_trend_forecast,
}


def evaluate_baselines(
    prices: np.ndarray,
    lookback: int,
    val_start_sample: int,
) -> Dict[str, Dict[str, float]]:
    """Score every baseline over the validation samples.

    Sample ``i`` uses ``prices[i:i + lookback]`` to predict
    ``prices[i + lookback]`` -- the same framing the LSTM dataset uses.

    Args:
        prices: Full price series in original units, oldest first.
        lookback: Sequence length.
        val_start_sample: Index of the first validation sample.

    Returns:
        ``{baseline_name: {mae, rmse, mape}}`` in original units.
    """
    from src.models.train import regression_metrics

    prices = np.asarray(prices, dtype=float).ravel()
    total_samples = len(prices) - lookback

    results: Dict[str, Dict[str, float]] = {}
    if total_samples <= val_start_sample:
        return results

    indices = range(val_start_sample, total_samples)
    targets = np.array([prices[i + lookback] for i in indices])

    for name, fn in BASELINES.items():
        predictions = np.array([fn(prices[i:i + lookback]) for i in indices])
        metrics = regression_metrics(targets, predictions)
        results[name] = {k: v for k, v in metrics.items() if k != "directional_accuracy"}

    return results
