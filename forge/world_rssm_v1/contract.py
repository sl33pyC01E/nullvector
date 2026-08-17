from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from ..config import PROJECT_ROOT


FORMAT = "nullvector-recurrent-world-student-v1/1.0.0"
CHECKPOINT_FORMAT = FORMAT + "-checkpoint"
REPORT_FORMAT = FORMAT + "-evaluation"
DEFAULT_CORPUS = PROJECT_ROOT / "outputs/world_action_cellular_v7/corpus_v1_6world"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/world_rssm_v1/production_v1"
SOURCE_FILES = (
    "forge/world_rssm_v1/__init__.py", "forge/world_rssm_v1/__main__.py",
    "forge/world_rssm_v1/contract.py", "forge/world_rssm_v1/model.py",
    "forge/world_rssm_v1/training.py",
)


@dataclass(frozen=True, slots=True)
class ModelConfig:
    hidden: int = 128
    condition: int = 96
    action_count: int = 22
    latent_channels: int = 48
    actor_features: int = 128


@dataclass(frozen=True, slots=True)
class TrainingPlan:
    total_updates: int = 1500
    segment_updates: int = 250
    batch_size: int = 2
    sequence: int = 8
    learning_rate: float = 2e-4
    ema_decay: float = .995
    seed: int = 0x5253534D5631

    def __post_init__(self):
        if self.total_updates % self.segment_updates or not 100 <= self.segment_updates <= self.total_updates <= 5000:
            raise ValueError("RSSM schedule drifted")
        if not 1 <= self.batch_size <= 8 or not 4 <= self.sequence <= 16:
            raise ValueError("RSSM batch geometry drifted")

    def to_dict(self):
        return asdict(self)


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def source_sha256() -> str:
    digest = hashlib.sha256(b"nullvector-recurrent-world-student-v1\0")
    for relative in SOURCE_FILES:
        digest.update(relative.encode() + b"\0" + (PROJECT_ROOT / relative).read_bytes() + b"\0")
    return digest.hexdigest()


def state_sha256(state: dict) -> str:
    digest = hashlib.sha256(b"nullvector-rssm-state-v1\0")
    for name, value in sorted(state.items()):
        digest.update(name.encode() + b"\0" + value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


import torch  # kept after constants so importing the contract remains cheap
