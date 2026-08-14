from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Final


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
PRIOR_FORMAT: Final[str] = "nullvector-neural-map-topology-masked-prior-smoke/1.0.0"
CHECKPOINT_FORMAT: Final[str] = "nullvector-neural-map-topology-masked-prior-checkpoint/1.0.0"
RAW_BANK_FORMAT: Final[str] = "nullvector-raw-neural-topology-latent-bank/1.0.0"
MASK_TOKEN: Final[int] = 512
CODEBOOK_SIZE: Final[int] = 512
FROZEN_CODEC_RELATIVE: Final[str] = (
    "outputs/map_topology_neural_production/calibration_500step_v2_hardened/checkpoint_final.pt"
)
FROZEN_CODEC_CHECKPOINT_SHA256: Final[str] = (
    "536d7e54e1da9f35ca9200353774121a59da69d9ea12853a5271b89fe06bce64"
)
FROZEN_CODEC_EMA_SHA256: Final[str] = (
    "0d90d210505fbda8fa3a319cc3a6d55ca1094252781159c4447f51bac6121d72"
)
FROZEN_CODEC_SOURCE_SHA256: Final[str] = (
    "1fe97d977aaf0a21e2caa6c75a52ee9a0e519087b8c2c2c1dca7e86806253a50"
)
SOURCE_FILES: Final[tuple[str, ...]] = (
    "forge/map_topology_neural_prior/__init__.py",
    "forge/map_topology_neural_prior/__main__.py",
    "forge/map_topology_neural_prior/checkpoint.py",
    "forge/map_topology_neural_prior/contract.py",
    "forge/map_topology_neural_prior/dataset.py",
    "forge/map_topology_neural_prior/masking.py",
    "forge/map_topology_neural_prior/model.py",
    "forge/map_topology_neural_prior/smoke.py",
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


def prior_source_sha256(root: Path = PROJECT_ROOT) -> str:
    return hashlib.sha256(canonical_json_bytes(source_manifest(root))).hexdigest()


@dataclass(frozen=True, slots=True)
class MaskedPriorConfig:
    width: int = 32
    residual_depth: int = 2
    steps: int = 2
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    gradient_clip: float = 1.0
    model_ema_decay: float = 0.99
    minimum_mask_fraction: float = 0.20
    maximum_mask_fraction: float = 0.90
    sampling_steps: int = 6
    seed: int = 0x4D41534B544F504F

    def __post_init__(self) -> None:
        if not 8 <= self.width <= 128 or not 0 <= self.residual_depth <= 6:
            raise ValueError("Masked-prior network dimensions are outside bounded foundation limits.")
        if isinstance(self.steps, bool) or not 1 <= self.steps <= 2:
            raise ValueError("Masked-prior CPU smoke is bounded to one or two steps.")
        values = (
            self.learning_rate,
            self.weight_decay,
            self.gradient_clip,
            self.model_ema_decay,
            self.minimum_mask_fraction,
            self.maximum_mask_fraction,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("Masked-prior floating configuration must be finite.")
        if not 0 < self.learning_rate <= 0.01 or not 0 <= self.weight_decay <= 1:
            raise ValueError("Masked-prior optimizer configuration is invalid.")
        if not 0 < self.gradient_clip <= 100 or not 0 <= self.model_ema_decay < 1:
            raise ValueError("Masked-prior clip/EMA configuration is invalid.")
        if not 0 < self.minimum_mask_fraction < self.maximum_mask_fraction <= 1:
            raise ValueError("Masked-prior mask fraction bounds are invalid.")
        if not 2 <= self.sampling_steps <= 32:
            raise ValueError("Masked-prior sampling steps must be in [2,32].")
        if isinstance(self.seed, bool) or not 0 <= self.seed < 1 << 63:
            raise ValueError("Masked-prior seed must be unsigned 63-bit.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MaskedPriorConfig":
        if not isinstance(payload, dict) or set(payload) != set(cls.__dataclass_fields__):
            raise ValueError("Masked-prior configuration members drifted.")
        return cls(**payload)

