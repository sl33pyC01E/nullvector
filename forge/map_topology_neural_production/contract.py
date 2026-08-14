from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Final

from ..map_topology_neural.codec import CodecConfig


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
CALIBRATION_FORMAT: Final[str] = "nullvector-neural-map-topology-codec-calibration/1.0.0"
CHECKPOINT_FORMAT: Final[str] = "nullvector-neural-map-topology-codec-production-checkpoint/1.0.0"
MIN_FREE_BYTES: Final[int] = 100 * 1024**3
MIN_FREE_CUDA_BYTES: Final[int] = 4 * 1024**3
CORE_SOURCE_FILES: Final[tuple[str, ...]] = (
    "forge/map_topology_neural/codec.py",
    "forge/map_topology_neural/contract.py",
    "forge/map_topology_neural/corpus.py",
    "forge/map_topology_neural/hashing.py",
)
PRODUCTION_SOURCE_FILES: Final[tuple[str, ...]] = (
    "forge/map_topology_neural_production/__init__.py",
    "forge/map_topology_neural_production/__main__.py",
    "forge/map_topology_neural_production/checkpoint.py",
    "forge/map_topology_neural_production/contract.py",
    "forge/map_topology_neural_production/dataset.py",
    "forge/map_topology_neural_production/metrics.py",
    "forge/map_topology_neural_production/training.py",
)


def canonical_json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def production_source_manifest(root: Path = PROJECT_ROOT) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in (*CORE_SOURCE_FILES, *PRODUCTION_SOURCE_FILES):
        path = Path(root) / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        result[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def production_source_sha256(root: Path = PROJECT_ROOT) -> str:
    return hashlib.sha256(canonical_json_bytes(production_source_manifest(root))).hexdigest()


@dataclass(frozen=True, slots=True)
class TopologyCodecCalibrationConfig:
    steps: int = 100
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    gradient_clip: float = 1.0
    model_ema_decay: float = 0.995
    seed: int = 0x544F504F43414C
    cell_budget: int = 131_072
    maximum_batch_size: int = 8
    validation_samples: int = 48
    test_samples: int = 24
    codec_width: int = 64
    latent_dim: int = 64
    codebook_size: int = 512
    field_embedding_dim: int = 8
    residual_depth: int = 2
    codebook_ema_decay: float = 0.99

    def __post_init__(self) -> None:
        if isinstance(self.steps, bool) or not 1 <= self.steps <= 500:
            raise ValueError("Topology codec calibration steps must be in [1,500].")
        finite = (self.learning_rate, self.weight_decay, self.gradient_clip, self.model_ema_decay)
        if any(not math.isfinite(value) for value in finite):
            raise ValueError("Topology codec calibration optimizer values must be finite.")
        if not 0 < self.learning_rate <= 0.01 or not 0 <= self.weight_decay <= 1:
            raise ValueError("Topology codec calibration optimizer values are outside safe bounds.")
        if not 0 < self.gradient_clip <= 100 or not 0 <= self.model_ema_decay < 1:
            raise ValueError("Topology codec calibration clip/EMA values are invalid.")
        if isinstance(self.seed, bool) or not 0 <= self.seed < 1 << 63:
            raise ValueError("Topology codec calibration seed must be unsigned 63-bit.")
        if not 32_768 <= self.cell_budget <= 1_048_576 or not 1 <= self.maximum_batch_size <= 32:
            raise ValueError("Topology codec calibration batch bounds are invalid.")
        if not 6 <= self.validation_samples <= 576 or not 6 <= self.test_samples <= 24:
            raise ValueError("Topology codec calibration evaluation census is invalid.")
        self.codec_config()

    def codec_config(self) -> CodecConfig:
        return CodecConfig(
            width=self.codec_width,
            latent_dim=self.latent_dim,
            codebook_size=self.codebook_size,
            field_embedding_dim=self.field_embedding_dim,
            residual_depth=self.residual_depth,
            ema_decay=self.codebook_ema_decay,
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["codec"] = self.codec_config().to_dict()
        return result

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TopologyCodecCalibrationConfig":
        if not isinstance(payload, dict):
            raise TypeError("Topology codec calibration config must be a mapping.")
        values = dict(payload)
        codec = values.pop("codec", None)
        if set(values) != set(asdict(cls())):
            raise ValueError("Topology codec calibration config members drifted.")
        config = cls(**values)
        if codec != config.codec_config().to_dict():
            raise ValueError("Topology codec calibration codec metadata drifted.")
        return config


QUALITY_GATES: Final[dict[str, float]] = {
    "minimum_loss_improvement": 0.02,
    "minimum_codebook_utilization": 0.03,
    "minimum_terrain_accuracy": 0.70,
    "minimum_hazard_macro_recall": 0.20,
    "minimum_elevation_accuracy": 0.45,
    "minimum_walkability_iou": 0.65,
}

