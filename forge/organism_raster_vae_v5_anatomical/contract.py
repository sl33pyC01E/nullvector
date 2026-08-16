from __future__ import annotations

from dataclasses import dataclass


FORMAT = "nullvector-anatomical-graph-raster-vae-v5/1.0.0"
CHECKPOINT_FORMAT = "nullvector-anatomical-graph-raster-vae-v5-checkpoint/1.0.0"
TOKEN_CONTRACT = "appendage-joint-organ-cell-authority-v1"

MAX_APPENDAGES = 8
MAX_JOINTS = 32
MAX_ORGANS = 8
MAX_TOKENS = MAX_APPENDAGES + MAX_JOINTS + MAX_ORGANS
TOKEN_FEATURES = 72

TOKEN_APPENDAGE = 0
TOKEN_JOINT = 1
TOKEN_ORGAN = 2

ORGAN_VOCAB = (
    "none", "brain", "phase_brain", "processor", "meristem", "heart",
    "vascular", "coolant_pump", "lung", "gut", "transmuter", "eye",
    "photoreceptor", "singularity", "optic", "bulb", "battery", "orbital",
    "jaw",
)


@dataclass(frozen=True, slots=True)
class TrainingPlan:
    segment_steps: int = 100
    batch_size: int = 8
    learning_rate: float = 8e-5
    ema_decay: float = .997
    seed: int = 0x414E41544F4D5935

    def __post_init__(self) -> None:
        if not 20 <= self.segment_steps <= 2000:
            raise ValueError("anatomical segment length drifted")
        if not 1 <= self.batch_size <= 64:
            raise ValueError("anatomical batch size drifted")
        if not 0 < self.learning_rate <= 1e-3:
            raise ValueError("anatomical learning rate drifted")
        if not .9 <= self.ema_decay < 1:
            raise ValueError("anatomical EMA drifted")
