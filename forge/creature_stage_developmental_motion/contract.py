from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Final

from ..config import PROJECT_ROOT
from ..creature_stage_developmental.contract import source_sha256 as developmental_source_sha256
from ..creature_stage_neural_motion_rollout.contract import source_sha256 as rollout_source_sha256


FORMAT: Final[str] = "nullvector-creature-stage-developmental-motion-corpus-v2"
SMOKE_FORMAT: Final[str] = "nullvector-creature-stage-developmental-actuator-smoke-v2"
PRODUCTION_FORMAT: Final[str] = "nullvector-creature-stage-developmental-actuator-production-v1"
CHECKPOINT_FORMAT: Final[str] = "nullvector-creature-stage-developmental-actuator-checkpoint-v1"
EVALUATION_FORMAT: Final[str] = "nullvector-creature-stage-developmental-actuator-evaluation-v1"

# Padding is a checkpoint interface.  Leave headroom above the reviewed v7
# maxima (560 cells, 43 nodes, 60 muscles) for later component diffusion.
MAX_CELLS: Final[int] = 640
MAX_NODES: Final[int] = 64
MAX_MUSCLES: Final[int] = 80
MAX_APPENDAGES: Final[int] = 32
MUSCLE_FEATURES: Final[int] = 10
NODE_FEATURES: Final[int] = 8
FRAME_COUNT: Final[int] = 72
FIXED_HZ: Final[int] = 12
MAX_DISPLACEMENT: Final[float] = 12.0

APPROVAL: Final[dict[str, object]] = {
    "status": "user-approved",
    "training_permitted": True,
    "scope": "centered-grasper-morphology-and-locomotion-authority-v7",
}

DEFAULT_REVIEW: Final[Path] = PROJECT_ROOT / "outputs/creature_stage_developmental/review_v7"
DEFAULT_CORPUS: Final[Path] = PROJECT_ROOT / "outputs/creature_stage_developmental_motion/corpus_v7_final_authority"
DEFAULT_PARENT: Final[Path] = (
    PROJECT_ROOT
    / "outputs/creature_stage_neural_motion_rollout/production_v1/cell_motion_rollout_0001000.pt"
)
DEFAULT_PRIOR: Final[Path] = PROJECT_ROOT / "outputs/creature_stage_developmental_motion/rollout1000_prior_v2"
DEFAULT_OUTPUT: Final[Path] = PROJECT_ROOT / "outputs/creature_stage_developmental_motion/production_v1"

CORPUS_SCHEMA: Final[Path] = PROJECT_ROOT / "shared/schema/creature_stage_developmental_motion_corpus.schema.json"
SMOKE_SCHEMA: Final[Path] = PROJECT_ROOT / "shared/schema/creature_stage_developmental_actuator_smoke.schema.json"
PRODUCTION_SCHEMA: Final[Path] = PROJECT_ROOT / "shared/schema/creature_stage_developmental_actuator_production.schema.json"
EVALUATION_SCHEMA: Final[Path] = PROJECT_ROOT / "shared/schema/creature_stage_developmental_actuator_evaluation.schema.json"
PRIOR_SCHEMA: Final[Path] = PROJECT_ROOT / "shared/schema/creature_stage_developmental_parent_prior.schema.json"

CORPUS_SOURCE_FILES: Final[tuple[str, ...]] = (
    "forge/creature_stage_developmental_motion/compiler.py",
    "shared/schema/creature_stage_developmental_motion_corpus.schema.json",
)
MODEL_SOURCE_FILES: Final[tuple[str, ...]] = (
    "forge/creature_stage_developmental_motion/dataset.py",
    "forge/creature_stage_developmental_motion/model.py",
    "forge/creature_stage_developmental_motion/parent_prior.py",
    "forge/creature_stage_developmental_motion/training.py",
    "forge/creature_stage_developmental_motion/evaluation.py",
    "forge/creature_stage_developmental_motion/smoke.py",
    "shared/schema/creature_stage_developmental_actuator_smoke.schema.json",
    "shared/schema/creature_stage_developmental_actuator_production.schema.json",
    "shared/schema/creature_stage_developmental_actuator_evaluation.schema.json",
    "shared/schema/creature_stage_developmental_parent_prior.schema.json",
)


@dataclass(frozen=True)
class DevelopmentalActuatorConfig:
    width: int = 256
    depth: int = 5
    heads: int = 8
    feedforward_multiplier: int = 3
    condition_width: int = 256
    cell_width: int = 96
    cell_graph_blocks: int = 1
    dropout: float = 0.04

    def __post_init__(self) -> None:
        if not 96 <= self.width <= 768 or not 2 <= self.depth <= 16:
            raise ValueError("developmental actuator transformer size drifted")
        if not 2 <= self.heads <= 16 or self.width % self.heads:
            raise ValueError("developmental actuator attention geometry drifted")
        if not 2 <= self.feedforward_multiplier <= 8:
            raise ValueError("developmental actuator feedforward geometry drifted")
        if not 96 <= self.condition_width <= 768:
            raise ValueError("developmental actuator condition width drifted")
        if not 64 <= self.cell_width <= 384 or not 1 <= self.cell_graph_blocks <= 8:
            raise ValueError("developmental actuator cellular residual geometry drifted")
        if not 0.0 <= self.dropout <= 0.25:
            raise ValueError("developmental actuator dropout drifted")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class DevelopmentalTrainingConfig:
    sequence_frames: int = 12
    cell_position_weight: float = 1.0
    cell_velocity_weight: float = 0.45
    node_position_weight: float = 1.25
    node_velocity_weight: float = 0.55
    muscle_weight: float = 0.70
    bone_length_weight: float = 0.65
    appendage_weight: float = 0.60
    anti_copy_weight: float = 0.25
    acceleration_weight: float = 0.12
    seam_weight: float = 0.80
    parent_prior_weight: float = 0.08
    teacher_forcing_start: float = 0.75
    teacher_forcing_end: float = 0.10
    seam_quota_numerator: int = 1
    seam_quota_denominator: int = 3

    def __post_init__(self) -> None:
        if not 6 <= self.sequence_frames <= 24:
            raise ValueError("developmental actuator sequence length drifted")
        for name in (
            "cell_position_weight", "cell_velocity_weight", "node_position_weight",
            "node_velocity_weight", "muscle_weight", "bone_length_weight",
            "appendage_weight", "anti_copy_weight", "acceleration_weight",
            "seam_weight", "parent_prior_weight",
        ):
            if not 0.0 <= float(getattr(self, name)) <= 4.0:
                raise ValueError(f"developmental actuator {name} drifted")
        if not 0.0 <= self.teacher_forcing_end <= self.teacher_forcing_start <= 1.0:
            raise ValueError("developmental actuator teacher-forcing schedule drifted")
        if (
            type(self.seam_quota_numerator) is not int
            or type(self.seam_quota_denominator) is not int
            or not 1 <= self.seam_quota_numerator < self.seam_quota_denominator <= 16
        ):
            raise ValueError("developmental actuator seam quota drifted")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def corpus_source_sha256() -> str:
    payload = {
        "format": FORMAT,
        "developmental_source_sha256": developmental_source_sha256(),
        "bounds": {
            "max_cells": MAX_CELLS,
            "max_nodes": MAX_NODES,
            "max_muscles": MAX_MUSCLES,
            "max_appendages": MAX_APPENDAGES,
            "muscle_features": MUSCLE_FEATURES,
            "node_features": NODE_FEATURES,
            "frame_count": FRAME_COUNT,
            "fixed_hz": FIXED_HZ,
            "max_displacement": MAX_DISPLACEMENT,
        },
        "approval": APPROVAL,
    }
    digest = hashlib.sha256(b"nullvector-developmental-motion-corpus-source-v2\0")
    digest.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii"))
    for relative in CORPUS_SOURCE_FILES:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        digest.update(relative.encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def source_sha256() -> str:
    payload = {
        "format": PRODUCTION_FORMAT,
        "corpus_source_sha256": corpus_source_sha256(),
        "rollout_source_sha256": rollout_source_sha256(),
        "model": DevelopmentalActuatorConfig().to_dict(),
        "training": DevelopmentalTrainingConfig().to_dict(),
    }
    digest = hashlib.sha256(b"nullvector-developmental-actuator-model-source-v1\0")
    digest.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii"))
    for relative in MODEL_SOURCE_FILES:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        digest.update(relative.encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()
