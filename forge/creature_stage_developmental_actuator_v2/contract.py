from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Final

from ..config import PROJECT_ROOT
from ..creature_stage_developmental_motion.contract import source_sha256 as v1_source_sha256


PRODUCTION_FORMAT: Final[str] = "nullvector-developmental-muscle-causal-production-v2"
CHECKPOINT_FORMAT: Final[str] = "nullvector-developmental-muscle-causal-checkpoint-v2"
EVALUATION_FORMAT: Final[str] = "nullvector-developmental-muscle-causal-evaluation-v2"
SEED: Final[int] = 0x4D5553434C455632

DEFAULT_TEACHER: Final[Path] = PROJECT_ROOT / "outputs/creature_stage_developmental_motion/corpus_v7_final_authority"
DEFAULT_PRIOR: Final[Path] = PROJECT_ROOT / "outputs/creature_stage_developmental_motion/rollout1000_prior_v2"
DEFAULT_V1_OUTPUT: Final[Path] = PROJECT_ROOT / "outputs/creature_stage_developmental_motion/production_v3"
DEFAULT_V1_SEED: Final[Path] = DEFAULT_V1_OUTPUT / "developmental_actuator_0002000.pt"
DEFAULT_OUTPUT: Final[Path] = PROJECT_ROOT / "outputs/creature_stage_developmental_actuator_v2/production_v1"

PRODUCTION_SCHEMA: Final[Path] = PROJECT_ROOT / "shared/schema/creature_stage_developmental_actuator_v2_production.schema.json"
EVALUATION_SCHEMA: Final[Path] = PROJECT_ROOT / "shared/schema/creature_stage_developmental_actuator_v2_evaluation.schema.json"

SOURCE_FILES: Final[tuple[str, ...]] = (
    "forge/creature_stage_developmental_actuator_v2/contract.py",
    "forge/creature_stage_developmental_actuator_v2/model.py",
    "forge/creature_stage_developmental_actuator_v2/training.py",
    "forge/creature_stage_developmental_actuator_v2/evaluation.py",
    "shared/schema/creature_stage_developmental_actuator_v2_production.schema.json",
    "shared/schema/creature_stage_developmental_actuator_v2_evaluation.schema.json",
)


@dataclass(frozen=True)
class CausalActuatorConfig:
    width: int = 256
    depth: int = 5
    heads: int = 8
    feedforward_multiplier: int = 3
    condition_width: int = 256
    cell_width: int = 96
    cell_graph_blocks: int = 1
    dropout: float = 0.04
    initial_previous_muscle_gate: float = -2.25
    initial_force_gate: float = -1.40
    direct_force_scale: float = 0.16

    def __post_init__(self) -> None:
        if not 96 <= self.width <= 768 or not 2 <= self.depth <= 16:
            raise ValueError("causal actuator transformer size drifted")
        if not 2 <= self.heads <= 16 or self.width % self.heads:
            raise ValueError("causal actuator attention geometry drifted")
        if not 2 <= self.feedforward_multiplier <= 8:
            raise ValueError("causal actuator feedforward geometry drifted")
        if not 96 <= self.condition_width <= 768 or not 64 <= self.cell_width <= 384:
            raise ValueError("causal actuator latent geometry drifted")
        if not 1 <= self.cell_graph_blocks <= 8 or not 0.0 <= self.dropout <= .25:
            raise ValueError("causal actuator cell graph drifted")
        if not -6.0 <= self.initial_previous_muscle_gate <= 0.0:
            raise ValueError("causal actuator recurrent muscle gate drifted")
        if not -6.0 <= self.initial_force_gate <= 2.0 or not 0.01 <= self.direct_force_scale <= .5:
            raise ValueError("causal actuator force scale drifted")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class CausalTrainingConfig:
    sequence_frames: int = 24
    cell_position_weight: float = 1.0
    cell_velocity_weight: float = .45
    node_position_weight: float = 1.25
    node_velocity_weight: float = .55
    muscle_weight: float = 1.20
    bone_length_weight: float = .70
    appendage_weight: float = .70
    anti_copy_weight: float = .24
    acceleration_weight: float = .14
    seam_weight: float = .90
    parent_prior_weight: float = .06
    muscle_l1_weight: float = .65
    muscle_velocity_weight: float = .35
    muscle_force_weight: float = .18
    teacher_forcing_start: float = .18
    teacher_forcing_end: float = 0.0
    seam_quota_numerator: int = 1
    seam_quota_denominator: int = 3

    def __post_init__(self) -> None:
        if not 12 <= self.sequence_frames <= 36:
            raise ValueError("causal actuator sequence length drifted")
        for name, value in asdict(self).items():
            if name in {"sequence_frames", "seam_quota_numerator", "seam_quota_denominator"}:
                continue
            if not 0.0 <= float(value) <= 4.0:
                raise ValueError(f"causal actuator {name} drifted")
        if not 0.0 <= self.teacher_forcing_end <= self.teacher_forcing_start <= .25:
            raise ValueError("causal actuator teacher forcing escaped low-forcing regime")
        if not 1 <= self.seam_quota_numerator < self.seam_quota_denominator <= 16:
            raise ValueError("causal actuator seam quota drifted")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def source_sha256() -> str:
    payload = {
        "format": PRODUCTION_FORMAT,
        "v1_source_sha256": v1_source_sha256(),
        "model": CausalActuatorConfig().to_dict(),
        "training": CausalTrainingConfig().to_dict(),
    }
    digest = hashlib.sha256(b"nullvector-developmental-muscle-causal-source-v2\0")
    digest.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii"))
    for relative in SOURCE_FILES:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        digest.update(relative.encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()
