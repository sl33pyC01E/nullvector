from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Final

from ..config import PROJECT_ROOT


FORMAT: Final[str] = "nullvector-neural-grounded-cell-motion-refine-v2"
PARENT: Final[Path] = PROJECT_ROOT / "outputs/creature_stage_neural_grounded/production_v1/grounded_motion_0001600.pt"
PARENT_SHA256: Final[str] = "6a9ab465284b720fd4431d8f1531ef79b1ddb15b5d15b536759a535ac02d7c39"
PARENT_SOURCE_SHA256: Final[str] = "2ee3b1de3b23b1f10a7b5a617e7f30e770eebff82313f225cf026560aa6f4a39"
DEFAULT_OUTPUT: Final[Path] = PROJECT_ROOT / "outputs/creature_stage_neural_grounded_refine/production_v2"
SOURCE_FILES: Final[tuple[str, ...]] = (
    "forge/creature_stage_neural_grounded_refine/__init__.py",
    "forge/creature_stage_neural_grounded_refine/__main__.py",
    "forge/creature_stage_neural_grounded_refine/contract.py",
    "forge/creature_stage_neural_grounded_refine/training.py",
)


@dataclass(frozen=True, slots=True)
class RefineConfig:
    updates: int = 400
    batch_size: int = 10
    sequence_frames: int = 6
    learning_rate: float = 1.5e-5
    backbone_scale: float = .05
    ema_decay: float = .98
    seam_every: int = 2

    def __post_init__(self) -> None:
        if not 100 <= self.updates <= 4000 or self.batch_size != 10 or not 3 <= self.sequence_frames <= 12 or not 1 <= self.seam_every <= 8:
            raise ValueError("grounded refine schedule drifted")
        if not 0 < self.learning_rate <= 5e-4 or not 0 < self.backbone_scale <= 1 or not .9 <= self.ema_decay < 1:
            raise ValueError("grounded refine optimizer drifted")

    def to_dict(self) -> dict[str, int | float]: return asdict(self)


def source_sha256() -> str:
    digest = hashlib.sha256(b"nullvector-neural-grounded-refine-v2\0")
    digest.update(json.dumps({"format": FORMAT, "parent": PARENT_SHA256, "config": RefineConfig().to_dict()}, sort_keys=True, separators=(",", ":")).encode("ascii"))
    for relative in SOURCE_FILES:
        path = PROJECT_ROOT / relative
        if not path.is_file(): raise FileNotFoundError(relative)
        digest.update(relative.encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()
