from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Final

from ..config import PROJECT_ROOT


FORMAT: Final[str] = "nullvector-creature-stage-neural-motion-v1"
CHECKPOINT_FORMAT: Final[str] = "nullvector-creature-stage-neural-motion-checkpoint-v1"
DEFAULT_TEACHER: Final[Path] = PROJECT_ROOT / "outputs/creature_stage_motion_corpus_v1_final_a"
DEFAULT_OUTPUT: Final[Path] = PROJECT_ROOT / "outputs/creature_stage_neural_motion/production_v1"
MAX_CELLS: Final[int] = 384
STATIC_FEATURES: Final[int] = 61
STATE_FEATURES: Final[int] = 4
CONTROL_FEATURES: Final[int] = 9
OUTPUT_FEATURES: Final[int] = 4
MAX_DISPLACEMENT: Final[float] = 12.0
GENE_NAMES: Final[tuple[str, ...]] = (
    "width", "height", "asymmetry", "symmetry", "repair", "metabolism",
    "fertility", "bond_strength",
)
TISSUES: Final[tuple[str, ...]] = (
    "skin", "structure", "armor", "neural", "circulatory", "respiratory",
    "digestive", "sensor", "locomotor", "storage", "phase", "root", "weapon",
)
ORGANS: Final[tuple[str, ...]] = (
    "none", "brain", "heart", "lung", "gut", "eye", "root", "stem", "bulb",
    "frond", "runner", "meristem", "vascular", "photoreceptor", "core",
    "orbital", "phase_bond", "phase_brain", "singularity", "flux", "transmuter",
    "drive", "mast", "hardpoint", "processor", "coolant_pump", "radiator",
    "battery", "optic",
)
SOURCE_FILES: Final[tuple[str, ...]] = (
    "forge/creature_stage_neural_motion/contract.py",
    "forge/creature_stage_neural_motion/dataset.py",
    "forge/creature_stage_neural_motion/model.py",
    "forge/creature_stage_neural_motion/training.py",
)


@dataclass(frozen=True)
class CellularMotionTransformerConfig:
    width: int = 384
    depth: int = 10
    heads: int = 8
    feedforward_multiplier: int = 4
    condition_width: int = 384
    dropout: float = 0.05
    static_features: int = STATIC_FEATURES
    state_features: int = STATE_FEATURES
    output_features: int = OUTPUT_FEATURES

    def __post_init__(self) -> None:
        if not 48 <= self.width <= 768 or self.width % self.heads:
            raise ValueError("cellular motion transformer width/head contract drifted")
        if not 2 <= self.depth <= 20 or not 2 <= self.heads <= 16:
            raise ValueError("cellular motion transformer depth contract drifted")
        if not 2 <= self.feedforward_multiplier <= 8:
            raise ValueError("cellular motion transformer feedforward contract drifted")
        if not 96 <= self.condition_width <= 768:
            raise ValueError("cellular motion condition width drifted")
        if not 0.0 <= self.dropout <= 0.25:
            raise ValueError("cellular motion dropout drifted")
        if (
            self.static_features != STATIC_FEATURES
            or self.state_features != STATE_FEATURES
            or self.output_features != OUTPUT_FEATURES
        ):
            raise ValueError("cellular motion tensor interface drifted")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def source_sha256() -> str:
    contract = {
        "format": FORMAT,
        "max_cells": MAX_CELLS,
        "static_features": STATIC_FEATURES,
        "state_features": STATE_FEATURES,
        "control_features": CONTROL_FEATURES,
        "output_features": OUTPUT_FEATURES,
        "max_displacement": MAX_DISPLACEMENT,
        "genes": list(GENE_NAMES),
        "tissues": list(TISSUES),
        "organs": list(ORGANS),
        "split": {"train": [0, 1], "validation": [2], "test": [3]},
    }
    digest = hashlib.sha256(b"nullvector-creature-stage-neural-motion-source-v1\0")
    digest.update(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("ascii"))
    for relative in SOURCE_FILES:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        digest.update(relative.encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()
