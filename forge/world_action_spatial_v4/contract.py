from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

from ..config import PROJECT_ROOT

CHECKPOINT_FORMAT = "nullvector-spatial-world-action-checkpoint/4.0.0"
REPORT_FORMAT = "nullvector-spatial-world-action-training/4.0.0"
SOURCE_FILES = (
    "forge/world_action_step_v3/data.py",
    "forge/world_action_spatial_v4/data.py",
    "forge/world_action_spatial_v4/contract.py",
    "forge/world_action_spatial_v4/model.py",
    "forge/world_action_spatial_v4/runtime.py",
    "forge/world_action_spatial_v4/training.py",
)


@dataclass(frozen=True, slots=True)
class ModelConfig:
    width: int = 512
    layers: int = 8
    heads: int = 8
    patch: int = 4
    spatial_channels: int = 4


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    steps: int = 7500
    batch_size: int = 16
    learning_rate: float = 2e-4
    ema_decay: float = 0.999
    changed_weight: float = 4.0
    spatial_weight: float = 7.0
    edge_weight: float = 0.16
    pixel_weight: float = 0.7
    pixel_changed_weight: float = 7.0
    pixel_batch: int = 4
    contrastive_weight: float = 0.35
    contrastive_margin: float = 0.025
    contrastive_batch: int = 8
    input_noise: float = 0.005
    seed: int = 0x5350415449414C34


def canonical(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def config_dict(value) -> dict:
    return asdict(value)


def source_sha256() -> str:
    digest = hashlib.sha256(b"nullvector-spatial-world-action-v4\0")
    for relative in SOURCE_FILES:
        digest.update(relative.encode() + b"\0" + (PROJECT_ROOT / relative).read_bytes() + b"\0")
    return digest.hexdigest()
