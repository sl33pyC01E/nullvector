from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Final

from ..config import PROJECT_ROOT


FORMAT: Final[str] = "nullvector-organism-neural-latent-refiner/1.0.0"
CHECKPOINT_FORMAT: Final[str] = "nullvector-organism-neural-latent-refiner-checkpoint/1.0.0"
FLOW_OUTPUT: Final[Path] = PROJECT_ROOT / "outputs/organism_latent_flow/prior_v1"
FLOW_SOURCE_SHA256: Final[str] = "27f2178f2090f70d8cbcc0be4c4a94fc7f12134f7af00479ece16e1d0515b9b9"
FLOW_MANIFEST_SHA256: Final[str] = "f524104372db57cabb4fe4d2fe91b6a72da9f7f39481fa1b8038d38cb4a3854e"
FLOW_CHECKPOINT_SHA256: Final[str] = "5b0a8f24e71fbad496c3a4a5f38453eedd501b941529ff491c4d9f27568778f1"
SOURCE_FILES: Final[tuple[str, ...]] = (
    "forge/organism_latent_refiner/__init__.py", "forge/organism_latent_refiner/__main__.py",
    "forge/organism_latent_refiner/artifacts.py", "forge/organism_latent_refiner/contract.py",
    "forge/organism_latent_refiner/model.py", "forge/organism_latent_refiner/training.py",
)


def canonical_json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_manifest() -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in SOURCE_FILES:
        path = PROJECT_ROOT / relative
        if not path.is_file(): raise FileNotFoundError(relative)
        result[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def source_sha256() -> str:
    return hashlib.sha256(canonical_json_bytes(source_manifest())).hexdigest()


@dataclass(frozen=True, slots=True)
class OrganismRefinerConfig:
    coarse_width: int = 192
    fine_width: int = 128
    condition_dim: int = 192
    time_dim: int = 96
    depth: int = 4
    corruption_min: float = .025
    corruption_max: float = .65
    interpolation_probability: float = .55
    ema_decay: float = .999
    auxiliary_batch: int = 8

    def __post_init__(self) -> None:
        integers = (self.coarse_width, self.fine_width, self.condition_dim, self.time_dim, self.depth, self.auxiliary_batch)
        if any(type(value) is not int for value in integers) or not 64 <= self.coarse_width <= 384 or not 48 <= self.fine_width <= 256 or not 64 <= self.condition_dim <= 384 or self.time_dim < 32 or not 1 <= self.depth <= 10 or not 1 <= self.auxiliary_batch <= 32:
            raise ValueError("Organism refiner dimensions drifted.")
        for value in (self.corruption_min, self.corruption_max, self.interpolation_probability, self.ema_decay):
            if isinstance(value, bool) or not math.isfinite(value): raise ValueError("Organism refiner scalar drifted.")
        if not 0 < self.corruption_min < self.corruption_max <= 1 or not 0 <= self.interpolation_probability <= 1 or not .9 <= self.ema_decay < 1:
            raise ValueError("Organism refiner scalar bounds drifted.")

    def to_dict(self) -> dict[str, Any]: return asdict(self)
