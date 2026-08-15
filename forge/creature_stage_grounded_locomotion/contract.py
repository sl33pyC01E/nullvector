from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Final

from ..config import PROJECT_ROOT
from ..creature_stage_developmental.contract import source_sha256 as developmental_source_sha256


FORMAT: Final[str] = "nullvector-grounded-developmental-locomotion-v1"
FAMILIES: Final[tuple[str, ...]] = (
    "humanoid", "animalian", "plantlike", "anomaly", "machine",
)
LOCOMOTOR_MODES: Final[tuple[str, ...]] = (
    "passive", "step", "drag", "float", "wheel",
)
SCHEMA_PATH: Final[Path] = (
    PROJECT_ROOT / "shared/schema/creature_stage_grounded_locomotion_review.schema.json"
)
SOURCE_FILES: Final[tuple[str, ...]] = (
    "forge/creature_stage_grounded_locomotion/__init__.py",
    "forge/creature_stage_grounded_locomotion/__main__.py",
    "forge/creature_stage_grounded_locomotion/contract.py",
    "forge/creature_stage_grounded_locomotion/physics.py",
    "forge/creature_stage_grounded_locomotion/review.py",
    "shared/schema/creature_stage_grounded_locomotion_review.schema.json",
)


@dataclass(frozen=True, slots=True)
class GroundedLocomotionConfig:
    frame_count: int = 72
    settle_cycles: int = 12
    substeps: int = 2
    edge_iterations: int = 7
    step_stance_fraction: float = 0.58
    drag_stance_fraction: float = 0.76
    wheel_stance_fraction: float = 0.56
    step_traction: float = 0.34
    drag_traction: float = 0.12
    wheel_traction: float = 0.29
    float_drive: float = 0.025
    body_damping: float = 0.86
    node_damping: float = 0.72
    gravity: float = 0.035
    contact_compliance: float = 0.08
    maximum_body_speed: float = 0.55

    def __post_init__(self) -> None:
        if (
            type(self.frame_count) is not int or not 48 <= self.frame_count <= 144
            or type(self.settle_cycles) is not int or not 4 <= self.settle_cycles <= 32
            or type(self.substeps) is not int or not 1 <= self.substeps <= 4
            or type(self.edge_iterations) is not int or not 3 <= self.edge_iterations <= 12
        ):
            raise ValueError("grounded locomotion discrete configuration drifted")
        for name in (
            "step_stance_fraction", "drag_stance_fraction", "wheel_stance_fraction",
            "step_traction", "drag_traction", "wheel_traction", "float_drive",
            "body_damping", "node_damping", "gravity", "contact_compliance",
            "maximum_body_speed",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"grounded locomotion {name} drifted")
        if not (
            0.45 <= self.step_stance_fraction <= 0.72
            and 0.62 <= self.drag_stance_fraction <= 0.90
            and 0.45 <= self.wheel_stance_fraction <= 0.68
            and 0.05 <= self.maximum_body_speed <= 0.90
        ):
            raise ValueError("grounded locomotion gait envelope drifted")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def source_sha256() -> str:
    payload = {
        "format": FORMAT,
        "families": FAMILIES,
        "locomotor_modes": LOCOMOTOR_MODES,
        "config": GroundedLocomotionConfig().to_dict(),
        "developmental_source_sha256": developmental_source_sha256(),
    }
    digest = hashlib.sha256(b"nullvector-grounded-locomotion-source-v1\0")
    digest.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii"))
    for relative in SOURCE_FILES:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        digest.update(relative.encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()
