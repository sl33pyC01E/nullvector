from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Final

from ..config import PROJECT_ROOT


FORMAT: Final[str] = "nullvector-organism-neural-cellular-automaton-v1"
CHECKPOINT_FORMAT: Final[str] = "nullvector-organism-neural-cellular-automaton-checkpoint-v1"
CORPUS_FORMAT: Final[str] = "nullvector-organism-neural-cellular-automaton-corpus-v1"
CANVAS_SIZE: Final[int] = 48
STATIC_CHANNELS: Final[int] = 85
BOND_CHANNELS: Final[int] = 8
DYNAMIC_NAMES: Final[tuple[str, ...]] = (
    "health", "fluid", "nutrient", "energy", "oxygen", "clot",
    "scar", "wound", "neural_activity", "surface_fluid", "biomass", "alive",
)
DYNAMIC_CHANNELS: Final[int] = len(DYNAMIC_NAMES)
DIRECTION_XY: Final[tuple[tuple[int, int], ...]] = (
    (-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1),
)

ANATOMY_MANIFEST: Final[Path] = PROJECT_ROOT / "outputs/cellular_breeding_symmetry_v1/cellular_symmetry_manifest.json"
PHYSIOLOGY_MANIFEST: Final[Path] = PROJECT_ROOT / "outputs/cellular_physiology_v4/cellular_physiology_manifest.json"
TRAUMA_MANIFEST: Final[Path] = PROJECT_ROOT / "outputs/cellular_trauma_v4/cellular_trauma_manifest.json"
DEFAULT_OUTPUT: Final[Path] = PROJECT_ROOT / "outputs/cellular_nca/nca_v1"

SOURCE_FILES: Final[tuple[str, ...]] = (
    "forge/cellular_nca/__init__.py",
    "forge/cellular_nca/__main__.py",
    "forge/cellular_nca/contract.py",
    "forge/cellular_nca/corpus.py",
    "forge/cellular_nca/model.py",
    "forge/cellular_nca/teacher.py",
    "forge/cellular_nca/training.py",
    "forge/cellular_nca/evaluation.py",
    "shared/schema/cellular_nca_manifest.schema.json",
)


@dataclass(frozen=True, slots=True)
class CellularNCAConfig:
    width: int = 256
    depth: int = 10
    expansion: int = 2
    max_delta: float = 0.16
    ema_decay: float = 0.999

    def __post_init__(self) -> None:
        if self.width % 32 or not 64 <= self.width <= 384 or not 4 <= self.depth <= 16:
            raise ValueError("Cellular NCA capacity contract drifted.")
        if self.expansion not in (1, 2, 3, 4) or not 0.01 <= self.max_delta <= 0.5:
            raise ValueError("Cellular NCA update contract drifted.")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_sha256() -> str:
    digest = hashlib.sha256(b"nullvector-organism-neural-cellular-automaton-source-v1\0")
    for relative in SOURCE_FILES:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        digest.update(relative.encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()
