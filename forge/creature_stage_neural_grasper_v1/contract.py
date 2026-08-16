from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path

from ..config import PROJECT_ROOT


FORMAT = "nullvector-neural-grounded-grasper-training/1.0.0"
CHECKPOINT_FORMAT = "nullvector-neural-grounded-grasper-checkpoint/1.0.0"
PARENT_CHECKPOINT = PROJECT_ROOT / "outputs/creature_stage_neural_grounded_controller/pilot_0800/grounded_controller_0000800.pt"
PARENT_SHA256 = "74ff3019b2f1d5886088a7034ef674da25defa2748501f78e876541723603acd"
MAX_APPENDAGES = 8
OWNER_FEATURES = 16
TARGET_FEATURES = 18
GLOBAL_FEATURES = 8
TARGET_TYPES = ("none", "organism", "object", "material")
GOALS = ("inspect", "consume", "carry", "tear", "throw")
SOURCE_FILES = (
    "forge/creature_stage_neural_grasper_v1/contract.py",
    "forge/creature_stage_neural_grasper_v1/dataset.py",
    "forge/creature_stage_neural_grasper_v1/model.py",
    "forge/creature_stage_neural_grasper_v1/constraint.py",
    "forge/creature_stage_neural_grasper_v1/feeding.py",
    "forge/creature_stage_neural_grasper_v1/training.py",
    "forge/creature_stage_neural_grasper_v1/runtime.py",
)


@dataclass(frozen=True, slots=True)
class ModelConfig:
    width: int = 256
    depth: int = 5
    dropout: float = 0.03


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    steps: int = 1800
    batch_size: int = 128
    learning_rate: float = 2e-4
    ema_decay: float = 0.997
    seed: int = 0x47524153504552


def config_dict(value) -> dict:
    return asdict(value)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_sha256() -> str:
    if file_sha256(PARENT_CHECKPOINT) != PARENT_SHA256:
        raise ValueError("grasper locomotion parent drifted")
    digest = hashlib.sha256(b"nullvector-neural-grounded-grasper-v1\0")
    digest.update(PARENT_SHA256.encode() + b"\0")
    for relative in SOURCE_FILES:
        digest.update(relative.encode() + b"\0" + (PROJECT_ROOT / relative).read_bytes() + b"\0")
    return digest.hexdigest()
