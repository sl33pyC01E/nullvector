from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Final

from ..config import PROJECT_ROOT
from ..creature_stage_developmental_actuator_v2.contract import source_sha256 as v2_source_sha256


PRODUCTION_FORMAT: Final[str] = "nullvector-developmental-length-projected-production-v3"
CHECKPOINT_FORMAT: Final[str] = "nullvector-developmental-length-projected-checkpoint-v3"
EVALUATION_FORMAT: Final[str] = "nullvector-developmental-length-projected-evaluation-v3"
SEED: Final[int] = 0x424F4E4550524A33

DEFAULT_V2_OUTPUT: Final[Path] = PROJECT_ROOT / "outputs/creature_stage_developmental_actuator_v2/production_v1"
DEFAULT_V2_SEED: Final[Path] = DEFAULT_V2_OUTPUT / "muscle_causal_actuator_0001200.pt"
DEFAULT_OUTPUT: Final[Path] = PROJECT_ROOT / "outputs/creature_stage_developmental_actuator_v3/production_v1"
PRODUCTION_SCHEMA: Final[Path] = PROJECT_ROOT / "shared/schema/creature_stage_developmental_actuator_v3_production.schema.json"
EVALUATION_SCHEMA: Final[Path] = PROJECT_ROOT / "shared/schema/creature_stage_developmental_actuator_v3_evaluation.schema.json"

SOURCE_FILES: Final[tuple[str, ...]] = (
    "forge/creature_stage_developmental_actuator_v3/contract.py",
    "forge/creature_stage_developmental_actuator_v3/model.py",
    "forge/creature_stage_developmental_actuator_v3/training.py",
    "forge/creature_stage_developmental_actuator_v3/evaluation.py",
    "shared/schema/creature_stage_developmental_actuator_v3_production.schema.json",
    "shared/schema/creature_stage_developmental_actuator_v3_evaluation.schema.json",
)


@dataclass(frozen=True)
class BoneProjectionConfig:
    iterations: int = 2
    relaxation: float = .92

    def __post_init__(self) -> None:
        if not 1 <= self.iterations <= 8 or not .25 <= self.relaxation <= 1.0:
            raise ValueError("length projection contract drifted")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def source_sha256() -> str:
    payload = {
        "format": PRODUCTION_FORMAT,
        "v2_source_sha256": v2_source_sha256(),
        "projection": BoneProjectionConfig().to_dict(),
    }
    digest = hashlib.sha256(b"nullvector-developmental-length-projected-source-v3\0")
    digest.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii"))
    for relative in SOURCE_FILES:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        digest.update(relative.encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()
