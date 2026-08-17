from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config import PROJECT_ROOT


FORMAT = "nullvector-current-anatomical-raster-vae-v6/1.0.0"
CHECKPOINT_FORMAT = "nullvector-current-anatomical-raster-vae-v6-checkpoint/1.0.0"
PARENT_CHECKPOINT = PROJECT_ROOT / "outputs/organism_raster_vae_v5_anatomical/production_002_hierarchical/segment_006/checkpoint.pt"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/organism_raster_vae_v6_current/production_v1"
SOURCE_FILES = (
    "forge/organism_raster_vae_v6_current/__init__.py",
    "forge/organism_raster_vae_v6_current/__main__.py",
    "forge/organism_raster_vae_v6_current/contract.py",
    "forge/organism_raster_vae_v6_current/training.py",
)


@dataclass(frozen=True, slots=True)
class TrainingPlan:
    segment_steps: int = 100
    batch_size: int = 8
    learning_rate: float = 6e-5
    ema_decay: float = .997
    seed: int = 0x43555252454E5436

    def __post_init__(self) -> None:
        if not 20 <= self.segment_steps <= 500:
            raise ValueError("V6 segment length drifted")
        if not 1 <= self.batch_size <= 16:
            raise ValueError("V6 batch size drifted")
        if not 0 < self.learning_rate <= 2e-4:
            raise ValueError("V6 learning rate drifted")
