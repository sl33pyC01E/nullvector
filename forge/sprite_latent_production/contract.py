from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from math import isfinite
from pathlib import Path
from typing import Any

from ..sprite_latent.codec import SpriteLatentConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FORMAT = "nullvector-semantic-sprite-fsq-production-v1"
CHECKPOINT_FORMAT = "nullvector-semantic-sprite-fsq-production-checkpoint-v1"
SEGMENT_FORMAT = "nullvector-semantic-sprite-fsq-production-segment-v1"
SUPERVISOR_FORMAT = "nullvector-semantic-sprite-fsq-production-supervisor-v1"
CALIBRATION_FORMAT = "nullvector-semantic-sprite-fsq-production-calibration-v1"
DEFAULT_CORPUS = PROJECT_ROOT / "data" / "morphology_32768_4d4f5250.npz"
DEFAULT_OUTPUT = PROJECT_ROOT / "checkpoints" / "sprite_latent_production_v1"
MIN_FREE_BYTES = 100 * 1024**3
MIN_FREE_CUDA_BYTES = 4 * 1024**3

CORE_SOURCE_FILES = (
    "forge/sprite_latent/codec.py",
    "forge/sprite_latent/corpus.py",
    "forge/sprite_latent/training.py",
)
PRODUCTION_SOURCE_FILES = (
    "forge/sprite_latent_production/__init__.py",
    "forge/sprite_latent_production/__main__.py",
    "forge/sprite_latent_production/checkpoint.py",
    "forge/sprite_latent_production/contract.py",
    "forge/sprite_latent_production/evaluation.py",
    "forge/sprite_latent_production/supervisor.py",
    "forge/sprite_latent_production/worker.py",
)


def canonical_json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def production_source_hash() -> str:
    digest = hashlib.sha256()
    digest.update(b"nullvector-sprite-fsq-production-source-v1\0")
    for relative in (*CORE_SOURCE_FILES, *PRODUCTION_SOURCE_FILES):
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        digest.update(relative.encode()); digest.update(b"\0"); digest.update(path.read_bytes()); digest.update(b"\0")
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ProductionConfig:
    epochs: int = 24
    segment_epochs: int = 2
    batch_size: int = 128
    evaluation_batch_size: int = 192
    learning_rate: float = 5.0e-4
    minimum_learning_rate: float = 2.5e-5
    weight_decay: float = 1.0e-4
    warmup_steps: int = 250
    continuous_warmup_epochs: int = 2
    gradient_clip: float = 1.0
    ema_decay: float = 0.999
    seed: int = 0x535052495445
    width: int = 96
    residual_depth: int = 3
    condition_dim: int = 96
    latent_levels: tuple[int, ...] = (8, 8, 6, 5, 5, 4)
    max_attempts: int = 3
    worker_timeout_seconds: int = 1800

    def __post_init__(self) -> None:
        if self.epochs < 2 or self.epochs > 1000 or self.segment_epochs < 1 or self.epochs % self.segment_epochs:
            raise ValueError("epochs must be a positive multiple of segment_epochs")
        if self.batch_size < 8 or self.batch_size > 4096 or self.evaluation_batch_size < 8 or self.evaluation_batch_size > 4096:
            raise ValueError("sprite latent production batches are too small")
        if not all(isfinite(value) for value in (self.learning_rate, self.minimum_learning_rate, self.weight_decay, self.gradient_clip, self.ema_decay)):
            raise ValueError("production optimizer values must be finite")
        if not 0.0 < self.minimum_learning_rate <= self.learning_rate or self.weight_decay < 0.0:
            raise ValueError("learning rates are invalid")
        if not 0.0 < self.ema_decay < 1.0 or self.gradient_clip <= 0.0:
            raise ValueError("EMA/gradient clip contract is invalid")
        if self.warmup_steps < 0 or self.continuous_warmup_epochs < 0 or self.continuous_warmup_epochs >= self.epochs:
            raise ValueError("continuous warmup must end before production training")
        if self.max_attempts not in (1, 2, 3) or self.worker_timeout_seconds < 60:
            raise ValueError("worker retry/timeout contract is invalid")
        if self.seed < 0 or self.seed >= 2**63:
            raise ValueError("production seed must be an unsigned 63-bit integer")
        self.codec_config()

    def codec_config(self) -> SpriteLatentConfig:
        return SpriteLatentConfig(
            width=self.width,
            latent_levels=self.latent_levels,
            residual_depth=self.residual_depth,
            condition_dim=self.condition_dim,
        )

    def metadata(self) -> dict[str, Any]:
        result = asdict(self)
        result["latent_levels"] = list(self.latent_levels)
        result["codec"] = self.codec_config().metadata()
        return result

    @classmethod
    def from_metadata(cls, payload: dict[str, Any]) -> "ProductionConfig":
        if not isinstance(payload, dict):
            raise TypeError("production config metadata must be a mapping")
        expected = set(asdict(cls())) | {"codec"}
        if set(payload) != expected:
            raise ValueError(
                f"production config metadata key mismatch missing={sorted(expected-set(payload))} "
                f"extra={sorted(set(payload)-expected)}"
            )
        values = dict(payload)
        codec = values.pop("codec")
        values["latent_levels"] = tuple(map(int, values["latent_levels"]))
        config = cls(**values)
        if codec != config.codec_config().metadata():
            raise ValueError("production config codec metadata mismatch")
        return config


QUALITY_GATES = {
    "aligned_tuple_accuracy": 0.93,
    "visible_tuple_accuracy": 0.78,
    "visible_silhouette_iou": 0.90,
    "minimum_family_visible_tuple_accuracy": 0.70,
    "minimum_family_visible_silhouette_iou": 0.84,
    "code_utilization": 0.015,
}
