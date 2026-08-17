from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

from ..config import PROJECT_ROOT


FORMAT = "nullvector-mobile-coordinator-distillation/1.0.0"
CORPUS_FORMAT = "nullvector-mobile-coordinator-corpus/1.0.0"
CHECKPOINT_FORMAT = "nullvector-mobile-coordinator-checkpoint/1.0.0"
PATCH = 32
MACRO_CHANNELS = 32
GLOBAL_FEATURES = 44
MEMBERS = 16
MEMBER_FEATURES = 64
TIMELINE = 24
TIMELINE_FEATURES = 64
ACTIONS = 5
SOURCE_FILES = (
    "forge/mobile_coordinator_student_v1/contract.py",
    "forge/mobile_coordinator_student_v1/model.py",
    "forge/mobile_coordinator_student_v1/corpus.py",
    "forge/mobile_coordinator_student_v1/training.py",
)


@dataclass(frozen=True, slots=True)
class ModelConfig:
    shared_width: int = 256
    macro_width: int = 64
    macro_blocks: int = 4
    member_width: int = 192


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    steps: int = 3600
    batch_size: int = 24
    learning_rate: float = 3e-4
    weight_decay: float = 2e-4
    ema_decay: float = .995
    seed: int = 0x434F4F52444D4F42


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def config_dict(value: object) -> dict[str, object]:
    return asdict(value)


def source_sha256() -> str:
    digest = hashlib.sha256(b"nullvector-mobile-coordinator-v1\0")
    for relative in SOURCE_FILES:
        digest.update(relative.encode() + b"\0" + (PROJECT_ROOT / relative).read_bytes() + b"\0")
    return digest.hexdigest()


def corpus_source_sha256() -> str:
    """Hash only the teacher projection contract, not mutable student training code."""
    digest = hashlib.sha256(canonical({
        "format": CORPUS_FORMAT,
        "patch": PATCH,
        "macro_channels": MACRO_CHANNELS,
        "global_features": GLOBAL_FEATURES,
        "members": MEMBERS,
        "member_features": MEMBER_FEATURES,
        "timeline": TIMELINE,
        "timeline_features": TIMELINE_FEATURES,
        "actions": ACTIONS,
    }))
    relative = "forge/mobile_coordinator_student_v1/corpus.py"
    digest.update(relative.encode() + b"\0" + (PROJECT_ROOT / relative).read_bytes())
    return digest.hexdigest()
