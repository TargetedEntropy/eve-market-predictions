"""Per-item model artifacts: checkpoint + metadata + scaler.

One shared ``best_model.pth`` meant whichever item trained last owned "the"
model, and the API served it for every ``type_id``. Artifacts are now keyed by
``(type_id, region_id)``:

``data/models/model_{type_id}_{region_id}.pth``
    ``state_dict`` only, loadable with ``weights_only=True``.
``data/models/model_{type_id}_{region_id}.json``
    Architecture, feature column order, MinMax scaler parameters, training
    metrics and the model version. JSON, so loading it never executes code.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

from src.config import settings
from src.models.lstm_model import MarketLSTM
from src.utils.timeutils import utcnow

logger = logging.getLogger(__name__)

ARTIFACT_FORMAT_VERSION = 2

#: (type_id, region_id, device) -> (checkpoint mtime, model, metadata)
_ARTIFACT_CACHE: Dict[Tuple[int, int, str], Tuple[int, MarketLSTM, "ModelMetadata"]] = {}


@dataclass
class ScalerParams:
    """MinMaxScaler state, stored as plain numbers rather than a pickle."""

    data_min: List[float]
    data_max: List[float]
    feature_range: Tuple[float, float] = (0.0, 1.0)

    @classmethod
    def from_scaler(cls, scaler: MinMaxScaler) -> "ScalerParams":
        return cls(
            data_min=[float(v) for v in scaler.data_min_],
            data_max=[float(v) for v in scaler.data_max_],
            feature_range=(float(scaler.feature_range[0]), float(scaler.feature_range[1])),
        )

    def to_scaler(self) -> MinMaxScaler:
        scaler = MinMaxScaler(feature_range=tuple(self.feature_range))
        data_min = np.asarray(self.data_min, dtype=float)
        data_max = np.asarray(self.data_max, dtype=float)
        # Reproduce exactly what MinMaxScaler.fit() derives, without refitting.
        data_range = data_max - data_min
        data_range[data_range == 0.0] = 1.0
        low, high = scaler.feature_range
        scaler.data_min_ = data_min
        scaler.data_max_ = data_max
        scaler.data_range_ = data_range
        scaler.scale_ = (high - low) / data_range
        scaler.min_ = low - data_min * scaler.scale_
        scaler.n_features_in_ = len(data_min)
        scaler.n_samples_seen_ = 1
        return scaler


@dataclass
class ModelMetadata:
    """Everything needed to reproduce inference for one item."""

    type_id: int
    region_id: int
    model_version: str
    trained_at: str
    lookback_window: int
    horizon_days: int
    feature_columns: List[str]
    input_size: int
    hidden_size: int
    num_layers: int
    dropout: float
    scaler: ScalerParams
    metrics: Dict[str, float] = field(default_factory=dict)
    baselines: Dict[str, Dict[str, float]] = field(default_factory=dict)
    n_train_samples: int = 0
    n_val_samples: int = 0
    train_end: Optional[str] = None
    format_version: int = ARTIFACT_FORMAT_VERSION

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["scaler"]["feature_range"] = list(self.scaler.feature_range)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":
        payload = dict(data)
        scaler = payload.pop("scaler")
        payload.pop("format_version", None)
        known = {f for f in cls.__dataclass_fields__ if f != "format_version"}
        payload = {k: v for k, v in payload.items() if k in known}
        return cls(
            scaler=ScalerParams(
                data_min=scaler["data_min"],
                data_max=scaler["data_max"],
                feature_range=tuple(scaler.get("feature_range", (0.0, 1.0))),
            ),
            **payload,
        )

    def build_scaler(self) -> MinMaxScaler:
        return self.scaler.to_scaler()


def model_dir() -> Path:
    return Path(settings.model_dir)


def checkpoint_path(type_id: int, region_id: int) -> Path:
    return model_dir() / f"model_{type_id}_{region_id}.pth"


def metadata_path(type_id: int, region_id: int) -> Path:
    return model_dir() / f"model_{type_id}_{region_id}.json"


def make_model_version(type_id: int, region_id: int, trained_at=None) -> str:
    stamp = (trained_at or utcnow()).strftime("%Y%m%d%H%M%S")
    return f"{type_id}-{region_id}-{stamp}"


def save_artifacts(
    state_dict: Dict[str, torch.Tensor],
    metadata: ModelMetadata,
) -> Tuple[Path, Path]:
    """Persist a checkpoint and its metadata side by side."""
    directory = model_dir()
    directory.mkdir(parents=True, exist_ok=True)

    ckpt = checkpoint_path(metadata.type_id, metadata.region_id)
    meta = metadata_path(metadata.type_id, metadata.region_id)

    torch.save({k: v.cpu() for k, v in state_dict.items()}, ckpt)
    meta.write_text(json.dumps(metadata.to_dict(), indent=2))

    logger.info(f"Saved model artifacts: {ckpt.name} (version {metadata.model_version})")
    return ckpt, meta


def load_metadata(type_id: int, region_id: int) -> Optional[ModelMetadata]:
    """Read metadata for an item, or None if the model has not been trained."""
    meta = metadata_path(type_id, region_id)
    if not meta.exists():
        return None
    try:
        return ModelMetadata.from_dict(json.loads(meta.read_text()))
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.error(f"Corrupt model metadata at {meta}: {e}")
        return None


def load_artifacts(
    type_id: int,
    region_id: int,
    device: Optional[torch.device] = None,
    use_cache: bool = True,
) -> Optional[Tuple[MarketLSTM, ModelMetadata]]:
    """Load a trained model and its metadata.

    ``weights_only=True`` keeps a tampered checkpoint in a mounted volume from
    executing arbitrary code at load time.

    Results are memoized on the checkpoint's mtime, so the API serves repeat
    requests without re-reading disk but still picks up a retrained model.
    """
    cache_key = (type_id, region_id, str(device))
    if use_cache:
        cached = _ARTIFACT_CACHE.get(cache_key)
        ckpt = checkpoint_path(type_id, region_id)
        if cached and ckpt.exists() and cached[0] == ckpt.stat().st_mtime_ns:
            return cached[1], cached[2]

    metadata = load_metadata(type_id, region_id)
    if metadata is None:
        return None

    ckpt = checkpoint_path(type_id, region_id)
    if not ckpt.exists():
        logger.error(f"Metadata present but checkpoint missing: {ckpt}")
        return None

    device = device or torch.device("cpu")
    state_dict = torch.load(ckpt, map_location=device, weights_only=True)

    model = MarketLSTM(
        input_size=metadata.input_size,
        hidden_size=metadata.hidden_size,
        num_layers=metadata.num_layers,
        dropout=metadata.dropout,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    _ARTIFACT_CACHE[cache_key] = (ckpt.stat().st_mtime_ns, model, metadata)

    return model, metadata


def clear_artifact_cache() -> None:
    """Drop memoized models (used by tests and after a retraining run)."""
    _ARTIFACT_CACHE.clear()


def available_models() -> List[Tuple[int, int]]:
    """List ``(type_id, region_id)`` pairs that have usable artifacts."""
    directory = model_dir()
    if not directory.exists():
        return []

    found = []
    for meta in sorted(directory.glob("model_*_*.json")):
        try:
            _, type_id, region_id = meta.stem.split("_")
            pair = (int(type_id), int(region_id))
        except ValueError:
            continue
        if checkpoint_path(*pair).exists():
            found.append(pair)
    return found
