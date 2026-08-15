from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Final

from ..config import PROJECT_ROOT
from ..creature_stage_neural_grounded_cyclic.contract import canonical_json_bytes, sha256_file


FORMAT: Final[str] = "nullvector-neural-grounded-controller-production-v1"
CHECKPOINT_FORMAT: Final[str] = "nullvector-neural-grounded-controller-checkpoint-v1"
EVALUATION_FORMAT: Final[str] = "nullvector-neural-grounded-controller-evaluation-v1"
PARENT: Final[Path] = PROJECT_ROOT / "outputs/creature_stage_neural_grounded_components/pilot_0450/component_grounded_motion_0001650.pt"
PARENT_SHA256: Final[str] = "fe7108b18ae3211632ddbb97566f24e664f241ad764884a55b90c1250d12b075"
MAX_APPENDAGES: Final[int] = 8
MAX_MUSCLES: Final[int] = 60
APPENDAGE_KINDS: Final[tuple[str, ...]] = ("arm", "leg", "tail", "root", "frond", "tendril", "wheel", "hardpoint")
OWNER_META_FEATURES: Final[int] = 16
MUSCLE_META_FEATURES: Final[int] = 8
SOURCE_FILES: Final[tuple[str, ...]] = (
    "forge/creature_stage_neural_grounded_controller/contract.py",
    "forge/creature_stage_neural_grounded_controller/dataset.py",
    "forge/creature_stage_neural_grounded_controller/model.py",
    "forge/creature_stage_neural_grounded_controller/physics.py",
    "forge/creature_stage_neural_grounded_controller/training.py",
)


@dataclass(frozen=True, slots=True)
class ControllerModelConfig:
    width: int = 256
    depth: int = 5
    dropout: float = .03

    def __post_init__(self) -> None:
        if not 128 <= self.width <= 512 or not 3 <= self.depth <= 8 or not 0 <= self.dropout <= .15:
            raise ValueError("grounded controller model contract drifted")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ControllerTrainingConfig:
    updates: int = 1600
    batch_size: int = 40
    learning_rate: float = 2e-4
    weight_decay: float = 1e-5
    ema_decay: float = .995

    def __post_init__(self) -> None:
        if not 100 <= self.updates <= 20_000 or self.batch_size < 10 or self.batch_size % 5:
            raise ValueError("grounded controller training shape drifted")
        if not 0 < self.learning_rate <= 1e-3 or not 0 <= self.weight_decay <= 1e-2 or not .9 <= self.ema_decay < 1:
            raise ValueError("grounded controller training scalar drifted")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def source_sha256() -> str:
    if sha256_file(PARENT) != PARENT_SHA256:
        raise ValueError("grounded controller parent bytes drifted")
    digest = hashlib.sha256(b"nullvector-neural-grounded-controller-source-v1\0")
    digest.update(PARENT_SHA256.encode("ascii") + b"\0")
    digest.update(canonical_json_bytes({
        "appendage_kinds": list(APPENDAGE_KINDS), "max_appendages": MAX_APPENDAGES,
        "max_muscles": MAX_MUSCLES, "owner_meta_features": OWNER_META_FEATURES,
        "muscle_meta_features": MUSCLE_META_FEATURES,
        "model": ControllerModelConfig().to_dict(), "training": ControllerTrainingConfig().to_dict(),
    }))
    for relative in SOURCE_FILES:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        digest.update(relative.encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()
