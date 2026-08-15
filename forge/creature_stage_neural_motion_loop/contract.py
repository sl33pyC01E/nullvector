from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Final

from ..config import PROJECT_ROOT
from ..creature_stage_neural_motion_rollout.contract import source_sha256 as parent_source_sha256


FORMAT: Final[str] = "nullvector-creature-stage-neural-motion-loop-v1"
DEFAULT_PARENT: Final[Path] = (
    PROJECT_ROOT
    / "outputs/creature_stage_neural_motion_rollout/production_v1/cell_motion_rollout_0001000.pt"
)
SOURCE_FILES: Final[tuple[str, ...]] = (
    "forge/creature_stage_neural_motion_loop/contract.py",
    "forge/creature_stage_neural_motion_loop/sampler.py",
    "forge/creature_stage_neural_motion_loop/smoke.py",
    "shared/schema/creature_stage_neural_motion_loop_smoke.schema.json",
)


@dataclass(frozen=True)
class LoopTrainingConfig:
    sequence_frames: int = 6
    appendage_weight: float = 0.40
    energy_weight: float = 0.10
    delta_weight: float = 1.00
    velocity_weight: float = 0.55
    graph_weight: float = 0.35
    minimum_energy_epsilon: float = 1e-4
    seam_quota_numerator: int = 1
    seam_quota_denominator: int = 4

    def __post_init__(self) -> None:
        if not 4 <= self.sequence_frames <= 12:
            raise ValueError("loop motion sequence length drifted")
        for name in (
            "appendage_weight", "energy_weight", "delta_weight",
            "velocity_weight", "graph_weight",
        ):
            if not 0.0 <= float(getattr(self, name)) <= 4.0:
                raise ValueError(f"loop motion {name} drifted")
        if not 1e-8 <= self.minimum_energy_epsilon <= 1e-2:
            raise ValueError("loop motion energy epsilon drifted")
        if (
            type(self.seam_quota_numerator) is not int
            or type(self.seam_quota_denominator) is not int
            or not 1 <= self.seam_quota_numerator < self.seam_quota_denominator <= 16
        ):
            raise ValueError("loop motion seam quota drifted")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def source_sha256() -> str:
    payload = {
        "format": FORMAT,
        "parent_rollout_source_sha256": parent_source_sha256(),
        "training": LoopTrainingConfig().to_dict(),
    }
    digest = hashlib.sha256(b"nullvector-cellular-motion-loop-source-v1\0")
    digest.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii"))
    for relative in SOURCE_FILES:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        digest.update(relative.encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()
