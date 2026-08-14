from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Final


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
PRIOR_V2_FORMAT: Final[str] = "nullvector-neural-map-topology-prior-v2-smoke/1.0.0"
CHECKPOINT_V2_FORMAT: Final[str] = "nullvector-neural-map-topology-prior-v2-state/1.0.0"
CODEBOOK_SIZE: Final[int] = 512
MASK_TOKEN: Final[int] = CODEBOOK_SIZE
SOURCE_FILES: Final[tuple[str, ...]] = (
    "forge/map_topology_neural_prior_v2/__init__.py",
    "forge/map_topology_neural_prior_v2/__main__.py",
    "forge/map_topology_neural_prior_v2/conditioning.py",
    "forge/map_topology_neural_prior_v2/contract.py",
    "forge/map_topology_neural_prior_v2/masking.py",
    "forge/map_topology_neural_prior_v2/model.py",
    "forge/map_topology_neural_prior_v2/smoke.py",
)


def canonical_json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_manifest(root: Path = PROJECT_ROOT) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in SOURCE_FILES:
        path = Path(root) / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        result[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def prior_v2_source_sha256(root: Path = PROJECT_ROOT) -> str:
    return hashlib.sha256(canonical_json_bytes(source_manifest(root))).hexdigest()


@dataclass(frozen=True, slots=True)
class PriorV2Config:
    width: int = 24
    levels: int = 3
    blocks_per_level: int = 1
    steps: int = 2
    learning_rate: float = 8.0e-4
    weight_decay: float = 1.0e-4
    gradient_clip: float = 1.0
    ema_decay: float = 0.99
    minimum_mask_fraction: float = 0.15
    maximum_mask_fraction: float = 0.98
    sampling_steps: int = 8
    seed: int = 0x5052494F525632

    def __post_init__(self) -> None:
        if type(self.width) is not int or not 8 <= self.width <= 96:
            raise ValueError("Prior-v2 width must be in [8,96].")
        if type(self.levels) is not int or not 2 <= self.levels <= 4:
            raise ValueError("Prior-v2 levels must be in [2,4].")
        if type(self.blocks_per_level) is not int or not 1 <= self.blocks_per_level <= 3:
            raise ValueError("Prior-v2 blocks per level must be in [1,3].")
        if type(self.steps) is not int or not 1 <= self.steps <= 4:
            raise ValueError("Prior-v2 foundation steps must be in [1,4].")
        floats = (
            self.learning_rate, self.weight_decay, self.gradient_clip, self.ema_decay,
            self.minimum_mask_fraction, self.maximum_mask_fraction,
        )
        if any(type(value) not in (int, float) or isinstance(value, bool) or not math.isfinite(value) for value in floats):
            raise ValueError("Prior-v2 floating configuration must be finite.")
        if not 0 < self.learning_rate <= 0.01 or not 0 <= self.weight_decay <= 1:
            raise ValueError("Prior-v2 optimizer configuration is invalid.")
        if not 0 < self.gradient_clip <= 100 or not 0 <= self.ema_decay < 1:
            raise ValueError("Prior-v2 clip/EMA configuration is invalid.")
        if not 0 < self.minimum_mask_fraction < self.maximum_mask_fraction <= 1:
            raise ValueError("Prior-v2 partial-mask bounds are invalid.")
        if type(self.sampling_steps) is not int or not 2 <= self.sampling_steps <= 32:
            raise ValueError("Prior-v2 sampling steps must be in [2,32].")
        if type(self.seed) is not int or not 0 <= self.seed < 1 << 63:
            raise ValueError("Prior-v2 seed must be unsigned 63-bit.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Any) -> "PriorV2Config":
        if not isinstance(payload, dict) or set(payload) != set(cls.__dataclass_fields__):
            raise ValueError("Prior-v2 configuration members drifted.")
        return cls(**payload)
