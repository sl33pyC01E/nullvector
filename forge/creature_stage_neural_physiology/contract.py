from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Final

from ..config import PROJECT_ROOT


FORMAT: Final[str] = "nullvector-creature-stage-neural-physiology-v1"
CHECKPOINT_FORMAT: Final[str] = "nullvector-creature-stage-neural-physiology-checkpoint-v1"
DEFAULT_TEACHER: Final[Path] = PROJECT_ROOT / "outputs/creature_stage_intervention_corpus_v1_final_a"
DEFAULT_OUTPUT: Final[Path] = PROJECT_ROOT / "outputs/creature_stage_neural_physiology/production_v1"
MAX_CELLS: Final[int] = 384
FLUID_SLOTS: Final[int] = 160
STATIC_FEATURES: Final[int] = 61
CELL_STATE_FEATURES: Final[int] = 4
SUMMARY_FEATURES: Final[int] = 10
FLUID_STATE_FEATURES: Final[int] = 7
EVENT_FEATURES: Final[int] = 4
CELL_POSITION_BOUND: Final[float] = 18.0
FLUID_SCALAR_BOUND: Final[float] = 8.0
SOURCE_FILES: Final[tuple[str, ...]] = (
    "forge/creature_stage_neural_physiology/contract.py",
    "forge/creature_stage_neural_physiology/dataset.py",
    "forge/creature_stage_neural_physiology/model.py",
    "forge/creature_stage_neural_physiology/training.py",
)


@dataclass(frozen=True)
class CellularPhysiologyTransformerConfig:
    width: int = 320
    depth: int = 8
    heads: int = 8
    feedforward_multiplier: int = 4
    condition_width: int = 320
    fluid_width: int = 192
    fluid_depth: int = 4
    dropout: float = 0.05
    static_features: int = STATIC_FEATURES
    cell_state_features: int = CELL_STATE_FEATURES
    summary_features: int = SUMMARY_FEATURES
    fluid_state_features: int = FLUID_STATE_FEATURES

    def __post_init__(self) -> None:
        if not 48 <= self.width <= 768 or self.width % self.heads:
            raise ValueError("cellular physiology width/head contract drifted")
        if not 2 <= self.depth <= 16 or not 2 <= self.heads <= 16:
            raise ValueError("cellular physiology depth contract drifted")
        if not 2 <= self.feedforward_multiplier <= 8:
            raise ValueError("cellular physiology feedforward contract drifted")
        if not 96 <= self.condition_width <= 768:
            raise ValueError("cellular physiology condition width drifted")
        if not 48 <= self.fluid_width <= 384 or not 1 <= self.fluid_depth <= 8:
            raise ValueError("cellular physiology fluid branch drifted")
        if not 0.0 <= self.dropout <= 0.25:
            raise ValueError("cellular physiology dropout drifted")
        if (
            self.static_features != STATIC_FEATURES
            or self.cell_state_features != CELL_STATE_FEATURES
            or self.summary_features != SUMMARY_FEATURES
            or self.fluid_state_features != FLUID_STATE_FEATURES
        ):
            raise ValueError("cellular physiology tensor interface drifted")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def source_sha256() -> str:
    contract = {
        "format": FORMAT,
        "max_cells": MAX_CELLS,
        "fluid_slots": FLUID_SLOTS,
        "static_features": STATIC_FEATURES,
        "cell_state_features": CELL_STATE_FEATURES,
        "summary_features": SUMMARY_FEATURES,
        "fluid_state_features": FLUID_STATE_FEATURES,
        "event_features": EVENT_FEATURES,
        "cell_position_bound": CELL_POSITION_BOUND,
        "fluid_scalar_bound": FLUID_SCALAR_BOUND,
        "split": {"train": [0, 1], "validation": [2], "test": [3]},
    }
    digest = hashlib.sha256(b"nullvector-creature-stage-neural-physiology-source-v1\0")
    digest.update(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("ascii"))
    for relative in SOURCE_FILES:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        digest.update(relative.encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()
