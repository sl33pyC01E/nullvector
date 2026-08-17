from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import torch

from ..config import PROJECT_ROOT

FORMAT = "nullvector-world-frame-rollout-refiner-v3/3.0.0"
CACHE_FORMAT = FORMAT + "-cache"
CHECKPOINT_FORMAT = FORMAT + "-checkpoint"
REPORT_FORMAT = FORMAT + "-report"
ROLLOUT_CORPUS = PROJECT_ROOT / "outputs/world_frame_rollout_decoder_v2/corpus_v1"
ROLLOUT_CORPUS_SHA256 = "97ec406ed32cb3bcb755e98b565af07b05129c1c511f4adc7ff06532788616cf"
DECODER = PROJECT_ROOT / "outputs/world_frame_rollout_decoder_v2/production_v1/runtime.pt"
DECODER_SHA256 = "68cf898b54948091b225156fa5c357072f81322c0001a4335365c1efb4ca01e2"
DEFAULT_CACHE = PROJECT_ROOT / "outputs/world_frame_rollout_refiner_v3/cache_v1"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/world_frame_rollout_refiner_v3/production_v1"
SOURCE_FILES = (
    "forge/world_frame_rollout_refiner_v3/__init__.py",
    "forge/world_frame_rollout_refiner_v3/__main__.py",
    "forge/world_frame_rollout_refiner_v3/contract.py",
    "forge/world_frame_rollout_refiner_v3/cache.py",
    "forge/world_frame_rollout_refiner_v3/training.py",
    "forge/world_frame_vae_refiner/model.py",
)
CACHE_SOURCE_FILES = (
    "forge/world_frame_rollout_refiner_v3/contract.py",
    "forge/world_frame_rollout_refiner_v3/cache.py",
    "forge/world_frame_vae/model.py",
)


@dataclass(frozen=True, slots=True)
class TrainingPlan:
    total_updates: int = 2400
    segment_updates: int = 600
    batch_size: int = 16
    crop: int = 128
    learning_rate: float = 1.5e-4
    ema_decay: float = 0.999
    identity_probability: float = 0.20
    seed: int = 0x524546494E455233

    def to_dict(self):
        return asdict(self)


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def file_sha256(path: Path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_sha256(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode() + b"\0" + value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def source_sha256():
    digest = hashlib.sha256(b"nullvector-world-frame-rollout-refiner-v3\0")
    for relative in SOURCE_FILES:
        digest.update(relative.encode() + b"\0" + (PROJECT_ROOT / relative).read_bytes() + b"\0")
    return digest.hexdigest()


def cache_source_sha256():
    digest = hashlib.sha256(b"nullvector-world-frame-rollout-refiner-v3-cache\0")
    for relative in CACHE_SOURCE_FILES:
        digest.update(relative.encode() + b"\0" + (PROJECT_ROOT / relative).read_bytes() + b"\0")
    return digest.hexdigest()
