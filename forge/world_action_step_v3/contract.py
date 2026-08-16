from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

from ..config import PROJECT_ROOT

CHECKPOINT_FORMAT = "nullvector-causal-world-action-step-checkpoint/3.0.0"
REPORT_FORMAT = "nullvector-causal-world-action-step-training/3.0.0"
SOURCE_FILES = (
    "forge/world_latent_dit/model.py",
    "forge/world_action_step_v3/contract.py",
    "forge/world_action_step_v3/data.py",
    "forge/world_action_step_v3/runtime.py",
    "forge/world_action_step_v3/training.py",
)


@dataclass(frozen=True, slots=True)
class ModelConfig:
    width: int = 512
    layers: int = 8
    heads: int = 8
    patch: int = 4


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    steps: int = 7500
    batch_size: int = 16
    learning_rate: float = 2e-4
    ema_decay: float = 0.9995
    changed_weight: float = 5.0
    edge_weight: float = 0.18
    input_noise: float = 0.006
    seed: int = 0x43415553414C5354


def canonical(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def config_dict(value) -> dict:
    return asdict(value)


def source_sha256() -> str:
    digest = hashlib.sha256(b"nullvector-causal-world-action-step-v3\0")
    for relative in SOURCE_FILES:
        digest.update(relative.encode() + b"\0" + (PROJECT_ROOT / relative).read_bytes() + b"\0")
    return digest.hexdigest()
