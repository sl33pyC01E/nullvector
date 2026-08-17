from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import torch

from ..config import PROJECT_ROOT

FORMAT = "nullvector-recurrent-world-student-v3/1.0.0"
CHECKPOINT_FORMAT = FORMAT + "-checkpoint"
REPORT_FORMAT = FORMAT + "-report"
ACTION_CHECKPOINT = PROJECT_ROOT / "outputs/recurrent_action_dit_v2/production_v1/runtime.pt"
ACTION_SHA256 = "22c3c9a23a411057bf376bee77b750318f459414f35bd4d87ab6e326b90323ef"
ACTOR_CHECKPOINT = PROJECT_ROOT / "outputs/actor_state_student_v1/production_v1/actor_0000800.pt"
ACTOR_FILE_SHA256 = "81738821946291b37e374c0e262b11f01298e9a50be627b90704ef52fa452158"
CORPUS = PROJECT_ROOT / "outputs/world_action_contiguous_v8/corpus_v1_6world"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/recurrent_world_student_v3/production_v1"
SOURCE_FILES = (
    "forge/recurrent_world_student_v3/__init__.py",
    "forge/recurrent_world_student_v3/__main__.py",
    "forge/recurrent_world_student_v3/contract.py",
    "forge/recurrent_world_student_v3/model.py",
    "forge/recurrent_world_student_v3/training.py",
)


@dataclass(frozen=True, slots=True)
class TrainingPlan:
    total_updates: int = 600
    segment_updates: int = 200
    rollout_steps: int = 4
    batch_size: int = 2
    learning_rate: float = 1e-5
    ema_decay: float = 0.999
    actor_weight: float = 0.35
    seed: int = 0x5245435552525633

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
    digest = hashlib.sha256(b"nullvector-recurrent-world-student-v3\0")
    for relative in SOURCE_FILES:
        digest.update(relative.encode() + b"\0" + (PROJECT_ROOT / relative).read_bytes() + b"\0")
    return digest.hexdigest()
