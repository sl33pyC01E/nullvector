from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Final

from ..config import PROJECT_ROOT
from ..creature_stage_neural_grounded_cyclic.contract import canonical_json_bytes


FORMAT: Final[str] = "nullvector-neural-grounded-components-production-v1"
CHECKPOINT_FORMAT: Final[str] = "nullvector-neural-grounded-components-checkpoint-v1"
MAX_APPENDAGES: Final[int] = 8
PARENT: Final[Path] = PROJECT_ROOT / "outputs/creature_stage_neural_grounded_cyclic/continuation_1200/cyclic_grounded_motion_0001200.pt"
PARENT_SHA256: Final[str] = "3357c7220089e48d04f8461bf1248e3e3063da9ca5d49ae808cca62ffa399d49"
SOURCE_FILES: Final[tuple[str, ...]] = (
    "forge/creature_stage_neural_grounded_components/contract.py",
    "forge/creature_stage_neural_grounded_components/dataset.py",
    "forge/creature_stage_neural_grounded_components/model.py",
    "forge/creature_stage_neural_grounded_components/training.py",
)


@dataclass(frozen=True, slots=True)
class ComponentModelConfig:
    width: int = 224
    depth: int = 4
    translation_scale: float = .10
    local_scale: float = .05

    def __post_init__(self) -> None:
        if not 128 <= self.width <= 384 or not 2 <= self.depth <= 6:
            raise ValueError("component motion model width/depth drifted")
        if not .02 <= self.translation_scale <= .20 or not .01 <= self.local_scale <= .10:
            raise ValueError("component motion correction scale drifted")

    def to_dict(self) -> dict[str, int | float]: return asdict(self)


@dataclass(frozen=True, slots=True)
class ComponentTrainingConfig:
    updates: int = 900
    batch_size: int = 5
    sequence_frames: int = 8
    learning_rate: float = 5e-5
    ema_decay: float = .99

    def __post_init__(self) -> None:
        if not 100 <= self.updates <= 5000 or self.batch_size < 5 or self.batch_size % 5 or not 3 <= self.sequence_frames <= 12:
            raise ValueError("component motion training shape drifted")
        if not 0 < self.learning_rate <= 1e-3 or not .9 <= self.ema_decay < 1:
            raise ValueError("component motion training scalar drifted")

    def to_dict(self) -> dict[str, int | float]: return asdict(self)


def source_sha256() -> str:
    digest = hashlib.sha256(b"nullvector-neural-grounded-components-source-v1\0" + PARENT_SHA256.encode("ascii"))
    digest.update(canonical_json_bytes({"model": ComponentModelConfig().to_dict(), "training": ComponentTrainingConfig().to_dict(), "max_appendages": MAX_APPENDAGES}))
    for relative in SOURCE_FILES:
        path = PROJECT_ROOT / relative
        digest.update(relative.encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()
