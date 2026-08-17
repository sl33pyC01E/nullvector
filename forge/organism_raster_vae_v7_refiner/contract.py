from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from ..config import PROJECT_ROOT


FORMAT = "nullvector-organism-raster-vae-v7-neural-refiner/1.0.0"
CACHE_FORMAT = "nullvector-organism-raster-vae-v7-cache/1.0.0"
CHECKPOINT_FORMAT = "nullvector-organism-raster-vae-v7-checkpoint/1.0.0"
MANIFEST_FORMAT = "nullvector-organism-raster-vae-v7-evaluation/1.0.0"
PARENT_CHECKPOINT = PROJECT_ROOT / "outputs/organism_raster_vae_v6_current/production_v2_b4/segment_005/checkpoint.pt"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/organism_raster_vae_v7_refiner/production_v1"
SOURCE_FILES = (
    "forge/organism_raster_vae_v7_refiner/__init__.py",
    "forge/organism_raster_vae_v7_refiner/__main__.py",
    "forge/organism_raster_vae_v7_refiner/contract.py",
    "forge/organism_raster_vae_v7_refiner/cache.py",
    "forge/organism_raster_vae_v7_refiner/model.py",
    "forge/organism_raster_vae_v7_refiner/training.py",
    "forge/organism_raster_vae_v7_refiner/evaluation.py",
)


@dataclass(frozen=True, slots=True)
class Plan:
    total_steps: int = 800
    segment_steps: int = 200
    batch_size: int = 16
    learning_rate: float = 2e-4
    ema_decay: float = .995
    seed: int = 0x524546494E455237

    def __post_init__(self) -> None:
        if self.total_steps % self.segment_steps or not 100 <= self.segment_steps <= self.total_steps <= 4000:
            raise ValueError("V7 schedule drifted")
        if not 4 <= self.batch_size <= 32 or not 0 < self.learning_rate <= 5e-4 or not .9 <= self.ema_decay < 1:
            raise ValueError("V7 optimizer contract drifted")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_sha256() -> str:
    digest = hashlib.sha256(b"nullvector-organism-raster-vae-v7-refiner\0")
    for relative in SOURCE_FILES:
        digest.update(relative.encode() + b"\0" + (PROJECT_ROOT / relative).read_bytes() + b"\0")
    digest.update(sha256_file(PARENT_CHECKPOINT).encode())
    return digest.hexdigest()
