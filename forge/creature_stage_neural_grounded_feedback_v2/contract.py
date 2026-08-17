from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib

from ..config import PROJECT_ROOT


FORMAT = "nullvector-neural-grounded-feedback-production/2.0.0"
CHECKPOINT_FORMAT = "nullvector-neural-grounded-feedback-checkpoint/2.0.0"
MAX_APPENDAGES = 8
MAX_MUSCLES = 60
OWNER_FEATURES = 23
GLOBAL_FEATURES = 23
MUSCLE_FEATURES = 8
APPENDAGE_KINDS = ("arm", "leg", "tail", "root", "frond", "tendril", "wheel", "hardpoint")
SOURCE_FILES = (
    "forge/creature_stage_neural_grounded_feedback_v2/contract.py",
    "forge/creature_stage_neural_grounded_feedback_v2/dataset.py",
    "forge/creature_stage_neural_grounded_feedback_v2/model.py",
    "forge/creature_stage_neural_grounded_feedback_v2/physics.py",
    "forge/creature_stage_neural_grounded_feedback_v2/runtime.py",
    "forge/creature_stage_neural_grounded_feedback_v2/training.py",
    "forge/creature_stage_developmental/development.py",
    "forge/creature_stage_developmental/motion.py",
    "forge/creature_stage_grounded_locomotion/physics.py",
)


@dataclass(frozen=True, slots=True)
class ModelConfig:
    width: int = 256
    depth: int = 6
    dropout: float = .025

    def __post_init__(self) -> None:
        if not 128 <= self.width <= 512 or not 3 <= self.depth <= 10 or not 0 <= self.dropout <= .15:
            raise ValueError("grounded feedback model contract drifted")

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    updates: int = 3000
    batch_size: int = 256
    learning_rate: float = 2.5e-4
    weight_decay: float = 1e-5
    ema_decay: float = .997
    variants_per_family: int = 2
    seed: int = 0x47524F554E444632

    def __post_init__(self) -> None:
        if not 100 <= self.updates <= 50_000 or not 64 <= self.batch_size <= 2048:
            raise ValueError("grounded feedback training shape drifted")
        if not 0 < self.learning_rate <= 2e-3 or not 0 <= self.weight_decay <= 1e-2 or not .9 <= self.ema_decay < 1:
            raise ValueError("grounded feedback training scalar drifted")
        if not 2 <= self.variants_per_family <= 12:
            raise ValueError("grounded feedback curriculum drifted")

    def to_dict(self):
        return asdict(self)


def source_sha256() -> str:
    digest = hashlib.sha256(b"nullvector-neural-grounded-feedback-source-v2\0")
    for relative in SOURCE_FILES:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        digest.update(relative.encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()
