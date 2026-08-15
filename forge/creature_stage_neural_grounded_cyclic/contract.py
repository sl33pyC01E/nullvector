from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Final

from ..config import PROJECT_ROOT
from ..creature_stage_grounded_locomotion.contract import source_sha256 as grounded_source_sha256


FORMAT: Final[str] = "nullvector-neural-grounded-cyclic-production-v1"
CHECKPOINT_FORMAT: Final[str] = "nullvector-neural-grounded-cyclic-checkpoint-v1"
EVALUATION_FORMAT: Final[str] = "nullvector-neural-grounded-cyclic-evaluation-v1"
MAX_CELLS: Final[int] = 560
STATIC_FEATURES: Final[int] = 61
STATE_FEATURES: Final[int] = 4
DYNAMIC_FEATURES: Final[int] = 16
POSITION_SCALE: Final[float] = 24.0
BODY_SPEED_SCALE: Final[float] = 0.55
TRAIN_IDENTITIES: Final[tuple[int, ...]] = (0, 2, 4, 6, 8)
EVALUATION_IDENTITIES: Final[tuple[int, ...]] = (1, 3, 5, 7, 9)
ROLLOUT_PARENT: Final[Path] = PROJECT_ROOT / "outputs/creature_stage_neural_motion_rollout/production_v1/cell_motion_rollout_0001000.pt"
ROLLOUT_PARENT_SHA256: Final[str] = "157a29eaee49221523ecc97dd7ba758461d3472930f9321f3918b1b4dd352513"
VAE_AUTHORITY: Final[Path] = PROJECT_ROOT / "outputs/organism_raster_vae_v2/fit_v2/checkpoint.pt"
DEFAULT_OUTPUT: Final[Path] = PROJECT_ROOT / "outputs/creature_stage_neural_grounded_cyclic/production_v1"
SOURCE_FILES: Final[tuple[str, ...]] = (
    "forge/creature_stage_neural_grounded_cyclic/contract.py",
    "forge/creature_stage_neural_grounded_cyclic/dataset.py",
    "forge/creature_stage_neural_grounded_cyclic/curriculum.py",
    "forge/creature_stage_neural_grounded_cyclic/model.py",
    "forge/creature_stage_neural_grounded_cyclic/training.py",
)


@dataclass(frozen=True, slots=True)
class CyclicModelConfig:
    refinement_width: int = 288
    refinement_depth: int = 5
    harmonics: int = 8
    direct_floor: float = 0.84
    recurrent_scale: float = 0.06

    def __post_init__(self) -> None:
        if not 128 <= self.refinement_width <= 512 or not 2 <= self.refinement_depth <= 8:
            raise ValueError("cyclic model width/depth drifted")
        if not 4 <= self.harmonics <= 16:
            raise ValueError("cyclic model harmonic count drifted")
        if not .75 <= self.direct_floor <= .98 or not .01 <= self.recurrent_scale <= .15:
            raise ValueError("cyclic model blend contract drifted")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CyclicTrainingConfig:
    updates: int = 2400
    batch_size: int = 5
    sequence_frames: int = 8
    learning_rate: float = 8e-5
    backbone_scale: float = .08
    weight_decay: float = 1e-5
    ema_decay: float = .99
    gradient_clip: float = 1.0
    seam_weight: float = .45

    def __post_init__(self) -> None:
        if not 100 <= self.updates <= 20_000 or self.batch_size < 5 or self.batch_size % 5:
            raise ValueError("cyclic training update/batch contract drifted")
        if not 3 <= self.sequence_frames <= 12:
            raise ValueError("cyclic training sequence length drifted")
        for name in ("learning_rate", "backbone_scale", "weight_decay", "ema_decay", "gradient_clip", "seam_weight"):
            value = float(getattr(self, name))
            if not 0 < value <= 2:
                raise ValueError(f"cyclic training {name} drifted")

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
        "grounded_source_sha256": grounded_source_sha256(),
        "rollout_parent_sha256": ROLLOUT_PARENT_SHA256,
        "train_identities": list(TRAIN_IDENTITIES),
        "evaluation_identities": list(EVALUATION_IDENTITIES),
        "runtime_honest_contacts": True,
        "model": CyclicModelConfig().to_dict(),
        "training": CyclicTrainingConfig().to_dict(),
    }
    digest = hashlib.sha256(b"nullvector-neural-grounded-cyclic-source-v1\0")
    digest.update(canonical_json_bytes(payload))
    for relative in SOURCE_FILES:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        digest.update(relative.encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()
