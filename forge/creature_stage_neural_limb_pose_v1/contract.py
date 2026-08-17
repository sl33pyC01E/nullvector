from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path

from ..config import PROJECT_ROOT


FORMAT = "nullvector-neural-limb-pose-production/1.0.0"
CHECKPOINT_FORMAT = "nullvector-neural-limb-pose-checkpoint/1.0.0"
MAX_NODES = 6
NODE_FEATURES = 8
CONTEXT_FEATURES = 21
APPENDAGE_KINDS = ("arm", "leg", "tail", "root", "frond", "tendril", "wheel", "hardpoint")
SOURCE_FILES = (
    "forge/creature_stage_neural_limb_pose_v1/contract.py",
    "forge/creature_stage_neural_limb_pose_v1/dataset.py",
    "forge/creature_stage_neural_limb_pose_v1/model.py",
    "forge/creature_stage_neural_limb_pose_v1/runtime.py",
    "forge/creature_stage_neural_limb_pose_v1/training.py",
    "forge/creature_stage_manipulation_v1/articulation.py",
    "forge/creature_stage_developmental/development.py",
    "forge/creature_stage_developmental/genomes.py",
    "forge/creature_stage_grounded_locomotion/physics.py",
)


@dataclass(frozen=True, slots=True)
class ModelConfig:
    width: int = 192
    depth: int = 5
    heads: int = 6
    dropout: float = .02

    def __post_init__(self) -> None:
        if not 96 <= self.width <= 512 or not 3 <= self.depth <= 10 or self.width % self.heads:
            raise ValueError("neural limb pose model shape drifted")
        if not 0 <= self.dropout <= .15:
            raise ValueError("neural limb pose dropout drifted")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    updates: int = 2400
    batch_size: int = 512
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    ema_decay: float = .997
    cases_per_appendage: int = 640
    seed: int = 0x4C494D42504F5345

    def __post_init__(self) -> None:
        if not 100 <= self.updates <= 50_000 or not 64 <= self.batch_size <= 4096:
            raise ValueError("neural limb pose training shape drifted")
        if not 0 < self.learning_rate <= 2e-3 or not 0 <= self.weight_decay <= 1e-2 or not .9 <= self.ema_decay < 1:
            raise ValueError("neural limb pose training scalar drifted")
        if not 64 <= self.cases_per_appendage <= 4096:
            raise ValueError("neural limb pose corpus census drifted")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_sha256() -> str:
    digest = hashlib.sha256(b"nullvector-neural-limb-pose-source-v1\0")
    for relative in SOURCE_FILES:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        digest.update(relative.encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()
