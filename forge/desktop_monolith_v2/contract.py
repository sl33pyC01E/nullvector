from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

from ..config import PROJECT_ROOT


FORMAT = "nullvector-desktop-world-monolith-v2/1.0.0"
CHECKPOINT_FORMAT = FORMAT + "-checkpoint"
ACTION_PARENT = PROJECT_ROOT / "outputs/recurrent_action_dit_v2/production_v1/runtime.pt"
ACTION_REPORT = PROJECT_ROOT / "outputs/recurrent_action_dit_v2/production_v1/report.json"
HIGH_CORPUS = PROJECT_ROOT / "outputs/mobile_coordinator_student_v1/corpus_desktop_001"
ACTION_CORPUS = PROJECT_ROOT / "outputs/world_action_cellular_v7/corpus_v1_6world"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/desktop_monolith_v2/production_001"
SOURCE_FILES = (
    "forge/desktop_monolith_v2/contract.py",
    "forge/desktop_monolith_v2/model.py",
    "forge/desktop_monolith_v2/training.py",
)


@dataclass(frozen=True, slots=True)
class ModelConfig:
    width: int = 512
    heads: int = 8
    fusion_layers: int = 4
    macro_patch: int = 4


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    steps: int = 3200
    batch_size: int = 6
    learning_rate: float = 1.5e-4
    weight_decay: float = 1e-3
    ema_decay: float = .997
    seed: int = 0x4445534B4D4F4E4F


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def config_dict(value: object) -> dict[str, object]:
    return asdict(value)


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_sha256() -> str:
    digest = hashlib.sha256(b"nullvector-desktop-world-monolith-v2\0")
    for relative in SOURCE_FILES:
        digest.update(relative.encode() + b"\0" + (PROJECT_ROOT / relative).read_bytes() + b"\0")
    return digest.hexdigest()
