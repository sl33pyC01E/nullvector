from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import torch

from ..config import PROJECT_ROOT


FORMAT = "nullvector-clean-recurrent-world-student-v4/1.0.0"
CHECKPOINT_FORMAT = FORMAT + "-checkpoint"
REPORT_FORMAT = FORMAT + "-evaluation"
PARENT = PROJECT_ROOT / "outputs/recurrent_world_student_v3/production_v1/runtime.pt"
PARENT_SHA256 = "6b6123ca21a1115819ea71bb082f95b14983592b14aecbdf7cf07c0819472411"
CORPUS = PROJECT_ROOT / "outputs/world_action_clean_v9/corpus_v1_6world"
CODEC = PROJECT_ROOT / "outputs/world_frame_decoder_adapt_v1/production_v1/runtime.pt"
CODEC_SHA256 = "8b29795559876ce9e067e3b7a1addd72911d0704af43612afd887e227f86947a"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/recurrent_world_student_v4/production_v1"
SOURCE_FILES = (
    "forge/recurrent_world_student_v4/__init__.py",
    "forge/recurrent_world_student_v4/__main__.py",
    "forge/recurrent_world_student_v4/contract.py",
    "forge/recurrent_world_student_v4/training.py",
    "forge/recurrent_world_student_v4/evaluation.py",
    "forge/recurrent_world_student_v3/model.py",
    "forge/recurrent_world_student_v3/training.py",
)


@dataclass(frozen=True, slots=True)
class TrainingPlan:
    total_updates: int = 400
    segment_updates: int = 200
    rollout_steps: int = 4
    batch_size: int = 2
    learning_rate: float = 7e-6
    ema_decay: float = .999
    actor_weight: float = .35
    seed: int = 0x434C45414E5634

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
    digest = hashlib.sha256(b"nullvector-clean-recurrent-world-student-v4\0")
    for relative in SOURCE_FILES:
        digest.update(relative.encode() + b"\0" + (PROJECT_ROOT / relative).read_bytes() + b"\0")
    return digest.hexdigest()
