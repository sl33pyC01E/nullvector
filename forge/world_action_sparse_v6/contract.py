from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

from ..config import PROJECT_ROOT

CHECKPOINT_FORMAT = "nullvector-sparse-world-action-checkpoint/6.0.0"
REPORT_FORMAT = "nullvector-sparse-world-action-training/6.0.0"
SOURCE_FILES = (
    "forge/world_action_step_v3/data.py",
    "forge/world_action_spatial_v4/data.py",
    "forge/world_action_sparse_v5/contract.py",
    "forge/world_action_sparse_v5/model.py",
    "forge/world_action_sparse_v5/training.py",
    "forge/world_action_sparse_v6/contract.py",
    "forge/world_action_sparse_v6/runtime.py",
    "forge/world_action_sparse_v6/training.py",
)


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    steps: int = 12000
    batch_size: int = 16
    learning_rate: float = 1e-4
    ema_decay: float = 0.999
    changed_weight: float = 7.0
    delta_weight: float = 0.75
    magnitude_weight: float = 0.4
    gate_weight: float = 0.35
    gate_positive_weight: float = 2.0
    leakage_weight: float = 0.45
    pixel_weight: float = 0.65
    pixel_changed_weight: float = 8.0
    pixel_batch: int = 4
    contrastive_weight: float = 0.22
    contrastive_margin: float = 0.012
    contrastive_batch: int = 8
    input_noise: float = 0.002
    validate_every: int = 1000
    seed: int = 0x5350415253455636


def canonical(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def config_dict(value) -> dict:
    return asdict(value)


def source_sha256() -> str:
    digest = hashlib.sha256(b"nullvector-sparse-world-action-v6\0")
    for relative in SOURCE_FILES:
        digest.update(relative.encode() + b"\0" + (PROJECT_ROOT / relative).read_bytes() + b"\0")
    return digest.hexdigest()
