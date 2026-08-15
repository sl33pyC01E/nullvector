from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Final

from ..config import PROJECT_ROOT
from ..creature_stage_grounded_locomotion.contract import source_sha256 as grounded_source_sha256


FORMAT: Final[str] = "nullvector-neural-grounded-cell-motion-v1"
CHECKPOINT_FORMAT: Final[str] = "nullvector-neural-grounded-cell-motion-checkpoint-v1"
MAX_CELLS: Final[int] = 560
DYNAMIC_FEATURES: Final[int] = 16
POSITION_SCALE: Final[float] = 24.0
BODY_SPEED_SCALE: Final[float] = 0.55
GROUND_AUTHORITY: Final[Path] = (
    PROJECT_ROOT / "outputs/creature_stage_grounded_locomotion/review_v1_final"
)
ROLLOUT_PARENT: Final[Path] = (
    PROJECT_ROOT
    / "outputs/creature_stage_neural_motion_rollout/production_v1/cell_motion_rollout_0001000.pt"
)
VAE_AUTHORITY: Final[Path] = (
    PROJECT_ROOT / "outputs/organism_raster_vae_v2/fit_v2/checkpoint.pt"
)
DEFAULT_OUTPUT: Final[Path] = PROJECT_ROOT / "outputs/creature_stage_neural_grounded/production_v1"
SOURCE_FILES: Final[tuple[str, ...]] = (
    "forge/creature_stage_neural_grounded/contract.py",
    "forge/creature_stage_neural_grounded/dataset.py",
    "forge/creature_stage_neural_grounded/model.py",
    "forge/creature_stage_neural_grounded/training.py",
)


@dataclass(frozen=True, slots=True)
class GroundedModelConfig:
    refinement_width: int = 256
    refinement_depth: int = 4
    dynamic_features: int = DYNAMIC_FEATURES

    def __post_init__(self) -> None:
        if (
            type(self.refinement_width) is not int
            or self.refinement_width < 96
            or self.refinement_width > 512
            or type(self.refinement_depth) is not int
            or not 2 <= self.refinement_depth <= 8
            or self.dynamic_features != DYNAMIC_FEATURES
        ):
            raise ValueError("grounded neural model configuration drifted")

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GroundedTrainingConfig:
    total_updates: int = 1600
    batch_size: int = 10
    sequence_frames: int = 6
    learning_rate: float = 1.2e-4
    backbone_learning_rate_scale: float = 0.18
    weight_decay: float = 1e-5
    ema_decay: float = 0.995
    gradient_clip: float = 1.0
    position_weight: float = 1.0
    velocity_weight: float = 0.35
    appendage_weight: float = 0.55
    contact_weight: float = 0.65
    graph_weight: float = 0.25
    body_velocity_weight: float = 0.55
    delta_weight: float = 0.20

    def __post_init__(self) -> None:
        if (
            type(self.total_updates) is not int or not 100 <= self.total_updates <= 20_000
            or type(self.batch_size) is not int or self.batch_size < 5 or self.batch_size % 5
            or type(self.sequence_frames) is not int or not 3 <= self.sequence_frames <= 12
        ):
            raise ValueError("grounded neural training schedule drifted")
        for name in (
            "learning_rate", "backbone_learning_rate_scale", "weight_decay",
            "ema_decay", "gradient_clip", "position_weight", "velocity_weight",
            "appendage_weight", "contact_weight", "graph_weight",
            "body_velocity_weight", "delta_weight",
        ):
            value = float(getattr(self, name))
            if not 0.0 < value <= 2.0:
                raise ValueError(f"grounded neural training {name} drifted")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_sha256() -> str:
    payload = {
        "format": FORMAT,
        "max_cells": MAX_CELLS,
        "dynamic_features": DYNAMIC_FEATURES,
        "position_scale": POSITION_SCALE,
        "body_speed_scale": BODY_SPEED_SCALE,
        "grounded_source_sha256": grounded_source_sha256(),
        "model": GroundedModelConfig().to_dict(),
        "training": GroundedTrainingConfig().to_dict(),
    }
    digest = hashlib.sha256(b"nullvector-neural-grounded-source-v1\0")
    digest.update(canonical_json_bytes(payload))
    for relative in SOURCE_FILES:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        digest.update(relative.encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()
