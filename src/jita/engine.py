"""Direct-horizon, chronological Jita forecasts. Nothing here uses regional history.

The neural candidate is a small return-regression MLP, retrained on each origin's
past only. Price scales and feature/target normalization are fitted anew per fold.
Naive persistence remains the production forecast unless the candidate beats it
on both earlier selection folds and later audit folds. Audit metrics of a rejected
candidate are retained; they are not presented as an untouched model-selection test.
"""
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from typing import Any

import numpy as np

from .schemas import Forecast, ItemForecast, Signals, Snapshot, TypeSource, Validation

ENGINE_VERSION = "jita-direct-mlp-v1"


@dataclass(frozen=True)
class EngineConfig:
    lookback: int = 14
    min_train_samples: int = 32
    epochs: int = 60
    hidden_units: int = 16
    learning_rate: float = 0.01
    seed: int = 1729
    max_folds: int = 32
    min_selection_samples: int = 6
    min_audit_samples: int = 6
    min_interval_samples: int = 20
    required_improvement: float = 0.02
    max_book_age_hours: float = 6.0
    entry_fee_pct: float = 0.0
    exit_fee_pct: float = 3.6
    slippage_pct: float = 0.5

    def __post_init__(self):
        if min(self.lookback, self.min_train_samples, self.epochs, self.hidden_units,
               self.max_folds, self.min_selection_samples, self.min_audit_samples) < 1:
            raise ValueError("training and validation counts must be positive")
        if self.min_interval_samples < 20:
            raise ValueError("at least 20 residuals required for empirical bands")
        if not 0 <= self.required_improvement < 1:
            raise ValueError("invalid required improvement")
        if self.learning_rate <= 0 or self.max_book_age_hours <= 0:
            raise ValueError("learning rate and freshness must be positive")
        if any(not math.isfinite(v) or not 0 <= v < 100 for v in
               (self.entry_fee_pct, self.exit_fee_pct, self.slippage_pct)):
            raise ValueError("fees must be finite percentages in [0,100)")


def resolve_device(require_cuda: bool = False) -> str:
    """Fail closed for GPU jobs, including wheels unable to execute on Blackwell."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    if not torch.cuda.is_available():
        if require_cuda:
            raise RuntimeError("CUDA required but unavailable; install a Blackwell-capable torch CUDA wheel")
        return "cpu"
    try:
        # A real kernel, not merely a driver-reported device. Old wheels can list
        # an RTX 5080 but fail their first operation on sm_120.
        x = torch.ones((8, 8), device="cuda")
        (x @ x).sum().item()
        torch.cuda.synchronize()
    except Exception as exc:
        if require_cuda:
            raise RuntimeError("CUDA kernel failed; a Blackwell-capable torch wheel is required") from exc
        return "cpu"
    return "cuda"


def make_training_arrays(prices: np.ndarray, horizon: int, lookback: int):
    """All labels end inside prices. Caller supplies only a historical prefix."""
    prices = np.asarray(prices, dtype=np.float64)
    returns = np.diff(np.log(prices))
    origins = range(lookback, len(prices) - horizon)
    x = np.array([returns[i-lookback:i] for i in origins], dtype=np.float32)
    y = np.array([math.log(prices[i+horizon] / prices[i]) for i in origins], dtype=np.float32)
    return x.reshape(-1, lookback), y


def train_predict(prices: np.ndarray, horizon: int, config: EngineConfig,
                  device: str = "cpu", inference_prices: np.ndarray | None = None
                  ) -> tuple[float, dict[str, Any]]:
    """Fit an actual torch model. Returns endpoint price and reproducible weights hash."""
    import torch
    x, y = make_training_arrays(prices, horizon, config.lookback)
    if len(y) < config.min_train_samples:
        raise ValueError("insufficient training labels")
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    if device == "cuda":
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
    xm, xs = x.mean(axis=0), x.std(axis=0).clip(1e-6)
    ym, ys = float(y.mean()), max(float(y.std()), 1e-6)
    xt = torch.tensor((x-xm)/xs, device=device)
    yt = torch.tensor((y-ym)/ys, device=device).reshape(-1, 1)
    model = torch.nn.Sequential(torch.nn.Linear(config.lookback, config.hidden_units),
                                torch.nn.Tanh(), torch.nn.Linear(config.hidden_units, 1)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    model.train()
    for _ in range(config.epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(xt), yt)
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite neural training loss")
        loss.backward()
        optimizer.step()
    inference = prices if inference_prices is None else inference_prices
    features = np.diff(np.log(inference))[-config.lookback:].astype(np.float32)
    model.eval()
    with torch.no_grad():
        log_return = float(model(torch.tensor((features-xm)/xs, device=device)).item()) * ys + ym
    # A bounded transform prevents numerical overflow; extrapolations are still
    # model estimates, not a promised profit or a probability.
    predicted = float(inference[-1] * math.exp(float(np.clip(log_return, -2, 2))))
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return predicted, {"weights_sha256": digest.hexdigest(), "training_samples": len(y),
                       "training_prices_sha256": hashlib.sha256(np.asarray(prices,dtype="<f8").tobytes()).hexdigest(),
                       "training_loss_standardized_mse": float(loss.item())}


def walk_forward(prices: np.ndarray, horizon: int, config: EngineConfig,
                 device: str = "cpu") -> list[dict[str, Any]]:
    """Non-overlapping horizon targets. Every model sees only prices[:origin+1]."""
    start = config.lookback + config.min_train_samples + horizon - 1
    origins = list(range(start, len(prices)-horizon, horizon))[-config.max_folds:]
    folds = []
    for origin in origins:
        pred, _ = train_predict(prices[:origin+1], horizon, config, device)
        folds.append({"origin": origin, "train_end": origin, "target": origin+horizon,
                      "actual": float(prices[origin+horizon]), "baseline": float(prices[origin]),
                      "candidate": pred})
    return folds


def baseline_folds(prices: np.ndarray, horizon: int, config: EngineConfig):
    return [{"origin": i, "target": i+horizon, "actual": float(prices[i+horizon]),
             "baseline": float(prices[i])}
            for i in list(range(config.lookback, len(prices)-horizon, horizon))[-config.max_folds:]]


def empirical_band(prediction: float, folds: list[dict], key: str, minimum: int):
    """Descriptive 10/90% historical relative-error quantiles, not calibrated odds."""
    if len(folds) < minimum:
        return None, None
    errors = np.array([f["actual"]/f[key]-1 for f in folds])
    lo, hi = np.quantile(errors, [0.1, 0.9])
    return max(0.0, float(prediction*(1+lo))), max(0.0, float(prediction*(1+hi)))


def net_return(predicted_mid: float, current_ask: float | None, spread_ratio: float | None,
               config: EngineConfig) -> float | None:
    if current_ask is None or spread_ratio is None:
        return None
    # Buy now at ask, sell later at estimated bid using today's relative spread.
    # Standing depth does not tell us how much can actually execute at that bid.
    future_bid = predicted_mid * (1-spread_ratio/2)
    proceeds = future_bid * (1-config.exit_fee_pct/100) * (1-config.slippage_pct/100)
    cost = current_ask * (1+config.entry_fee_pct/100)
    return (proceeds/cost-1)*100


class ForecastEngine:
    def __init__(self, config: EngineConfig | None = None, require_cuda: bool = False):
        self.config = config or EngineConfig()
        self.device = resolve_device(require_cuda)
        self.fit_metadata: dict[str, Any] = {}

    def forecast_item(self, source: TypeSource, now: datetime | None = None) -> ItemForecast:
        now = now or datetime.now(timezone.utc)
        if now.tzinfo is None:
            raise ValueError("now must be timezone aware")
        source = TypeSource.model_validate(source.model_dump())
        rows = sorted(source.history, key=lambda r: r.date)
        if len({r.date for r in rows}) != len(rows):
            raise ValueError("duplicate history dates")
        if any(r.date > now.date() for r in rows) or source.book.observed_at > now:
            raise ValueError("future source data")
        book = source.book
        if book.best_bid is None or book.best_ask is None:
            raise ValueError("two-sided station book required")
        current = (book.best_bid+book.best_ask)/2
        complete = [r for r in rows if r.date < now.date()]
        # Only a contiguous daily suffix ending yesterday can support calendar
        # forecasts. Never compress gaps into fake consecutive trading days.
        suffix = []
        expected = now.date()-timedelta(days=1)
        for row in reversed(complete):
            if row.date != expected:
                break
            suffix.append(row)
            expected -= timedelta(days=1)
        suffix.reverse()
        prices = np.array([r.mid_price for r in suffix], dtype=np.float64)
        fresh = (now-book.observed_at).total_seconds() <= self.config.max_book_age_hours*3600
        spread = (book.best_ask-book.best_bid)/current
        forecasts = []
        for horizon in (1, 7, 30):
            validation = Validation(status="insufficient_history", training_observations=len(prices),
                                    last_training_date=suffix[-1].date if suffix else None)
            pred, model_name = current, "naive_persistence"
            folds = baseline_folds(prices, horizon, self.config)
            candidate_folds = []
            # Avoid even fitting if there cannot be independent selection/audit folds.
            start = self.config.lookback+self.config.min_train_samples+horizon-1
            fold_count = len(range(start, max(start, len(prices)-horizon), horizon))
            if fresh and min(fold_count,self.config.max_folds) >= self.config.min_selection_samples+self.config.min_audit_samples:
                candidate_folds = walk_forward(prices, horizon, self.config, self.device)
            key = "baseline"
            if candidate_folds:
                cut = max(self.config.min_selection_samples, len(candidate_folds)//3)
                selection, audit = candidate_folds[:cut], candidate_folds[cut:]
                mae = lambda fs, k: float(np.mean([abs(f[k]-f["actual"]) for f in fs]))
                sb, sc = mae(selection, "baseline"), mae(selection, "candidate")
                ab, ac = mae(audit, "baseline"), mae(audit, "candidate")
                validation.selection_samples = len(selection)
                validation.selection_baseline_mae, validation.selection_model_mae = sb, sc
                validation.candidate_mae = ac
                folds = audit
                if sc < sb*(1-self.config.required_improvement) and ac < ab*(1-self.config.required_improvement):
                    # Current live midpoint is observed today, not a completed daily
                    # training label. Fit on completed data; predict using live features.
                    # Inference includes today's live price, retaining yesterday's
                    # training boundary via a separate feature suffix.
                    pred, metadata = self._live_predict(prices, current, horizon)
                    self.fit_metadata[f"{source.type_id}:{horizon}"] = metadata
                    model_name, key = "direct_return_mlp", "candidate"
                    validation.status = "candidate_passed_selection_and_audit"
                else:
                    validation.status = "baseline_candidate_not_better"
            elif folds:
                validation.status = "baseline_only_short_history"
            if not fresh:
                validation.status = "stale_book_baseline_only"
                folds = []
            lower, upper = empirical_band(pred, folds, key, self.config.min_interval_samples)
            if folds:
                validation.backtest_samples = len(folds)
                validation.baseline_mae = float(np.mean([abs(f["baseline"]-f["actual"]) for f in folds]))
                validation.model_mae = float(np.mean([abs(f[key]-f["actual"]) for f in folds]))
                validation.directional_accuracy = float(np.mean([
                    np.sign(f[key]-f["baseline"]) == np.sign(f["actual"]-f["baseline"]) for f in folds]))
            if lower is not None:
                validation.interval_samples = len(folds)
                validation.interval_method = "historical_nonoverlapping_relative_residual_quantiles_10_90_not_calibrated"
                # No calibrated coverage claim: historical quantiles alone do not establish it.
            forecasts.append(Forecast(horizon_days=horizon, target_date=now.date()+timedelta(days=horizon),
                predicted_price=pred, lower_price=lower, upper_price=upper,
                expected_return_pct=(pred/current-1)*100,
                net_return_pct=net_return(pred, book.best_ask, spread, self.config) if fresh else None,
                model_name=model_name, validation=validation))
        total = (book.buy_volume or 0)+(book.sell_volume or 0)
        known_depth = book.buy_volume is not None and book.sell_volume is not None
        signals = Signals(spread_pct=spread*100,
            book_imbalance=(book.buy_volume-book.sell_volume)/total if known_depth and total else None,
            standing_bid_depth=book.buy_volume, standing_ask_depth=book.sell_volume,
            fee_entry_pct=self.config.entry_fee_pct, fee_exit_pct=self.config.exit_fee_pct,
            slippage_pct=self.config.slippage_pct,
            # Executed liquidity is unknown: never assert an executable trade.
            actionable=False)
        return ItemForecast(type_id=source.type_id,item_name=source.item_name,
            market_group_id=source.market_group_id,market_group_name=source.market_group_name,
            current_price=current,book=book,history=rows[-90:],forecasts=forecasts,signals=signals,
            data_quality={"fresh_book":fresh,"complete_daily_observations":len(complete),
                "contiguous_completed_days":len(suffix),"gapped_history":len(suffix)<len(complete),
                "partial_today_excluded":any(r.date==now.date() for r in rows),
                "history_sha256":hashlib.sha256(json.dumps([r.model_dump(mode="json") for r in rows],
                    sort_keys=True,separators=(",",":")).encode()).hexdigest(),
                "source":"evemarket_station_hub_bid_ask_daily_midpoint",
                "historical_depth_is_executed_volume":False})

    def _live_predict(self, prices, current, horizon):
        # Shared fitting path with a distinct inference suffix; no live label enters training.
        return train_predict_live(prices, current, horizon, self.config, self.device)

    def snapshot(self, sources: list[TypeSource], now: datetime | None = None) -> Snapshot:
        import torch
        now = now or datetime.now(timezone.utc)
        self.fit_metadata = {}
        items = [self.forecast_item(source, now) for source in sources]
        config = asdict(self.config)
        digest = hashlib.sha256(json.dumps(config,sort_keys=True,separators=(",",":")).encode()).hexdigest()
        return Snapshot(run_id=now.strftime("%Y%m%dT%H%M%S%fZ"),generated_at=now,
            data_as_of=min(i.book.observed_at for i in items),items=items,
            model={"engine_version":ENGINE_VERSION,"config":config,"config_sha256":digest,
                   "torch_version":str(torch.__version__),"device":self.device,"fits":self.fit_metadata,
                   "cuda_version":torch.version.cuda,
                   "device_name":torch.cuda.get_device_name(0) if self.device == "cuda" else "cpu",
                   "selection_policy":"candidate must beat naive by required margin on selection and later audit; audit participates in fallback gate",
                   "interval_policy":"empirical historical residual bands, no coverage or profit probability guarantee"},
            quality={"item_count":len(items),"stale_items":sum(not i.data_quality["fresh_book"] for i in items),
                     "executed_volume_available":False})


def train_predict_live(prices, current, horizon, config, device):
    """Implemented by the same trainer with separate inference prices."""
    return train_predict(prices, horizon, config, device, inference_prices=np.append(prices, current))
