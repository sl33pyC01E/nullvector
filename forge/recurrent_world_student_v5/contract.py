from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import torch

from ..config import PROJECT_ROOT


FORMAT = "nullvector-perception-recurrent-world-student-v5/1.0.0"
CHECKPOINT_FORMAT = FORMAT + "-checkpoint"
REPORT_FORMAT = FORMAT + "-evaluation"
PARENT = PROJECT_ROOT / "outputs/recurrent_world_student_v4/production_v1/runtime.pt"
PARENT_SHA256 = "2555420fa81d864c1e648760a19bd34f70941f754f4a1a113e0f77f8ff02c8d8"
CORPUS = PROJECT_ROOT / "outputs/world_action_natural_v10/corpus_v1_6world"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/recurrent_world_student_v5/production_v1"
SOURCE_FILES = (
    "forge/recurrent_world_student_v5/__init__.py",
    "forge/recurrent_world_student_v5/__main__.py",
    "forge/recurrent_world_student_v5/contract.py",
    "forge/recurrent_world_student_v5/model.py",
    "forge/recurrent_world_student_v5/training.py",
    "forge/recurrent_world_student_v3/model.py",
)


@dataclass(frozen=True, slots=True)
class TrainingPlan:
    total_updates: int = 600
    segment_updates: int = 100
    rollout_steps: int = 4
    batch_size: int = 4
    learning_rate: float = 1e-5
    ema_decay: float = .999
    actor_weight: float = .35
    perception_dropout: float = .08
    seed: int = 0x5045524345505635

    def to_dict(self):
        return asdict(self)


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def file_sha256(path: Path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_sha256(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode() + b"\0" + value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def source_sha256():
    digest = hashlib.sha256(b"nullvector-perception-recurrent-world-student-v5\0")
    for relative in SOURCE_FILES:
        digest.update(relative.encode() + b"\0" + (PROJECT_ROOT / relative).read_bytes() + b"\0")
    return digest.hexdigest()
