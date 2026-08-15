from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Final

from ..config import PROJECT_ROOT
from ..creature_stage_neural_motion.contract import source_sha256 as parent_source_sha256


FORMAT: Final[str] = "nullvector-creature-stage-neural-motion-rollout-v1"
CHECKPOINT_FORMAT: Final[str] = "nullvector-creature-stage-neural-motion-rollout-checkpoint-v1"
DEFAULT_PARENT: Final[Path] = (
    PROJECT_ROOT
    / "outputs/creature_stage_neural_motion/production_v1/cell_motion_0003000.pt"
)
DEFAULT_OUTPUT: Final[Path] = (
    PROJECT_ROOT / "outputs/creature_stage_neural_motion_rollout/production_v1"
)
SOURCE_FILES: Final[tuple[str, ...]] = (
    "forge/creature_stage_neural_motion_rollout/contract.py",
    "forge/creature_stage_neural_motion_rollout/training.py",
    "shared/schema/creature_stage_neural_motion_rollout_smoke.schema.json",
)


@dataclass(frozen=True)
class RolloutTrainingConfig:
    sequence_frames: int = 4
    appendage_weight: float = 0.50
    energy_weight: float = 0.25
    delta_weight: float = 0.25
    velocity_weight: float = 0.45
    graph_weight: float = 0.30
    minimum_energy_epsilon: float = 1e-4

    def __post_init__(self) -> None:
        if not 2 <= self.sequence_frames <= 12:
            raise ValueError("rollout motion sequence length drifted")
        for name in (
            "appendage_weight",
            "energy_weight",
            "delta_weight",
            "velocity_weight",
            "graph_weight",
        ):
            if not 0.0 <= float(getattr(self, name)) <= 4.0:
                raise ValueError(f"rollout motion {name} drifted")
        if not 1e-8 <= self.minimum_energy_epsilon <= 1e-2:
            raise ValueError("rollout motion energy epsilon drifted")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def source_sha256() -> str:
    dependency = parent_source_sha256()
    payload = {
        "format": FORMAT,
        "parent_model_source_sha256": dependency,
        "training": RolloutTrainingConfig().to_dict(),
    }
    digest = hashlib.sha256(b"nullvector-cellular-motion-rollout-source-v1\0")
    digest.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii"))
    for relative in SOURCE_FILES:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        digest.update(relative.encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()
