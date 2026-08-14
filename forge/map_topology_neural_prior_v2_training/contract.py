from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Final

from ..map_topology_neural_prior_training.contract import (
    FROZEN_LATENT_CORPUS_IDENTITY_SHA256,
    FROZEN_LATENT_CORPUS_MANIFEST_FILE_SHA256,
    FROZEN_LATENT_CORPUS_RELATIVE,
)
from ..map_topology_neural_prior_v2.contract import PriorV2Config, prior_v2_source_sha256


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
SEGMENT_FORMAT: Final[str] = "nullvector-neural-map-topology-prior-v2-calibration-segment/1.0.0"
CHECKPOINT_FORMAT: Final[str] = "nullvector-neural-map-topology-prior-v2-calibration-checkpoint/1.0.0"
SOURCE_FILES: Final[tuple[str, ...]] = (
    "forge/map_topology_neural_prior_v2_training/__init__.py",
    "forge/map_topology_neural_prior_v2_training/__main__.py",
    "forge/map_topology_neural_prior_v2_training/checkpoint.py",
    "forge/map_topology_neural_prior_v2_training/contract.py",
    "forge/map_topology_neural_prior_v2_training/metrics.py",
    "forge/map_topology_neural_prior_v2_training/training.py",
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


def training_v2_source_sha256(root: Path = PROJECT_ROOT) -> str:
    return hashlib.sha256(canonical_json_bytes(source_manifest(root))).hexdigest()


@dataclass(frozen=True, slots=True)
class PriorV2CalibrationConfig:
    total_steps: int = 24
    steps_per_segment: int = 4
    width: int = 48
    levels: int = 3
    blocks_per_level: int = 2
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    gradient_clip: float = 1.0
    ema_decay: float = 0.995
    cell_budget: int = 32_768
    maximum_batch_size: int = 16
    minimum_mask_fraction: float = 0.15
    maximum_mask_fraction: float = 0.98
    sampling_steps: int = 8
    validation_samples: int = 48
    test_samples: int = 24
    seed: int = 0x5052494F52325632

    def __post_init__(self) -> None:
        ints = (self.total_steps, self.steps_per_segment, self.width, self.levels, self.blocks_per_level, self.cell_budget, self.maximum_batch_size, self.sampling_steps, self.validation_samples, self.test_samples, self.seed)
        if any(type(value) is not int for value in ints):
            raise ValueError("Prior-v2 calibration integer fields must be exact integers.")
        if not 1 <= self.total_steps <= 20_000 or not 1 <= self.steps_per_segment <= min(500, self.total_steps):
            raise ValueError("Prior-v2 calibration step bounds are invalid.")
        if not 16 <= self.width <= 96 or not 2 <= self.levels <= 4 or not 1 <= self.blocks_per_level <= 3:
            raise ValueError("Prior-v2 calibration model dimensions are invalid.")
        floats = (self.learning_rate, self.weight_decay, self.gradient_clip, self.ema_decay, self.minimum_mask_fraction, self.maximum_mask_fraction)
        if any(isinstance(value, bool) or not math.isfinite(value) for value in floats):
            raise ValueError("Prior-v2 calibration floating fields must be finite.")
        if not 0 < self.learning_rate <= .01 or not 0 <= self.weight_decay <= 1 or not 0 < self.gradient_clip <= 100 or not 0 <= self.ema_decay < 1:
            raise ValueError("Prior-v2 calibration optimizer bounds are invalid.")
        if not 0 < self.minimum_mask_fraction < self.maximum_mask_fraction <= 1:
            raise ValueError("Prior-v2 calibration mask bounds are invalid.")
        if not 4_096 <= self.cell_budget <= 262_144 or not 1 <= self.maximum_batch_size <= 64:
            raise ValueError("Prior-v2 calibration batch bounds are invalid.")
        if not 2 <= self.sampling_steps <= 32 or self.validation_samples not in (6, 48) or self.test_samples not in (6, 24):
            raise ValueError("Prior-v2 calibration evaluation bounds are invalid.")
        if not 0 <= self.seed < 1 << 63:
            raise ValueError("Prior-v2 calibration seed must be unsigned 63-bit.")

    def model_config(self) -> PriorV2Config:
        return PriorV2Config(
            width=self.width, levels=self.levels, blocks_per_level=self.blocks_per_level,
            steps=2, learning_rate=self.learning_rate, weight_decay=self.weight_decay,
            gradient_clip=self.gradient_clip, ema_decay=self.ema_decay,
            minimum_mask_fraction=self.minimum_mask_fraction,
            maximum_mask_fraction=self.maximum_mask_fraction,
            sampling_steps=self.sampling_steps, seed=self.seed,
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["model"] = self.model_config().to_dict()
        result["prior_v2_source_sha256"] = prior_v2_source_sha256()
        return result

    @classmethod
    def from_dict(cls, payload: Any) -> "PriorV2CalibrationConfig":
        if not isinstance(payload, dict):
            raise ValueError("Prior-v2 calibration config must be a mapping.")
        values = dict(payload)
        model = values.pop("model", None)
        prior_source = values.pop("prior_v2_source_sha256", None)
        if set(values) != set(asdict(cls())):
            raise ValueError("Prior-v2 calibration config members drifted.")
        config = cls(**values)
        if model != config.model_config().to_dict() or prior_source != prior_v2_source_sha256():
            raise ValueError("Prior-v2 calibration model/source binding drifted.")
        return config


FROZEN_AUTHORITY: Final[dict[str, str]] = {
    "latent_corpus_manifest_file_sha256": FROZEN_LATENT_CORPUS_MANIFEST_FILE_SHA256,
    "latent_corpus_identity_sha256": FROZEN_LATENT_CORPUS_IDENTITY_SHA256,
}
