from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

from ..config import PROJECT_ROOT

CHECKPOINT_FORMAT = "nullvector-sparse-world-action-checkpoint/5.0.0"
REPORT_FORMAT = "nullvector-sparse-world-action-training/5.0.0"
SOURCE_FILES = (
    "forge/world_action_step_v3/data.py",
    "forge/world_action_spatial_v4/data.py",
    "forge/world_action_sparse_v5/contract.py",
    "forge/world_action_sparse_v5/model.py",
    "forge/world_action_sparse_v5/runtime.py",
    "forge/world_action_sparse_v5/training.py",
)


@dataclass(frozen=True, slots=True)
class ModelConfig:
    width: int = 512
    layers: int = 8
    heads: int = 8
    patch: int = 4
    spatial_channels: int = 5
    gate_bias: float = -4.0


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    steps: int = 8000
    batch_size: int = 16
    learning_rate: float = 2e-4
    ema_decay: float = 0.999
    changed_weight: float = 9.0
    delta_weight: float = 0.8
    gate_weight: float = 0.55
    gate_positive_weight: float = 3.0
    leakage_weight: float = 0.3
    edge_weight: float = 0.12
    pixel_weight: float = 0.7
    pixel_changed_weight: float = 9.0
    pixel_batch: int = 4
    contrastive_weight: float = 0.28
    contrastive_margin: float = 0.018
    contrastive_batch: int = 8
    input_noise: float = 0.003
    seed: int = 0x5350415253455635


def canonical(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def config_dict(value) -> dict:
    return asdict(value)


def source_sha256() -> str:
    digest = hashlib.sha256(b"nullvector-sparse-world-action-v5\0")
    for relative in SOURCE_FILES:
        digest.update(relative.encode() + b"\0" + (PROJECT_ROOT / relative).read_bytes() + b"\0")
    return digest.hexdigest()
