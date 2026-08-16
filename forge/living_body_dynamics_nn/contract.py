from __future__ import annotations

from dataclasses import dataclass


FORMAT = "nullvector-living-body-graph-dynamics/1.0.0"
CHECKPOINT_FORMAT = "nullvector-living-body-graph-dynamics-checkpoint/1.0.0"
FEATURES = 39
SYSTEMS = 7


@dataclass(frozen=True, slots=True)
class DynamicsConfig:
    width: int = 256
    depth: int = 6
    family_width: int = 32
    dropout: float = .03

    def __post_init__(self) -> None:
        if not 128 <= self.width <= 768 or not 3 <= self.depth <= 12:
            raise ValueError("living dynamics geometry drifted")
        if not 16 <= self.family_width <= 128 or not 0 <= self.dropout <= .25:
            raise ValueError("living dynamics regularization drifted")


@dataclass(frozen=True, slots=True)
class TrainingPlan:
    segment_steps: int = 250
    batch_size: int = 8
    learning_rate: float = 2e-4
    ema_decay: float = .998
    seed: int = 0x4C4956494E474E4E
