from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import torch

from ..config import PROJECT_ROOT

FORMAT = "nullvector-world-frame-decoder-adapt-v1/1.0.0"
CHECKPOINT_FORMAT = FORMAT + "-checkpoint"
REPORT_FORMAT = FORMAT + "-report"
BASE_CHECKPOINT = PROJECT_ROOT / "outputs/world_frame_vae/production_v2_high_fidelity/checkpoint.pt"
BASE_SHA256 = "875691e4be9866000ea4a112ca708ccd0755fa98d3fdededa3bb09bf3b560259"
CELLULAR_CORPUS = PROJECT_ROOT / "outputs/world_action_cellular_v7/corpus_v1_6world"
ORIGINAL_EPISODES = tuple(PROJECT_ROOT / f"outputs/action_teacher_v1/curriculum-v1-{letter}" for letter in "abcdef")
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/world_frame_decoder_adapt_v1/production_v1"
SOURCE_FILES = (
    "forge/world_frame_decoder_adapt_v1/__init__.py",
    "forge/world_frame_decoder_adapt_v1/__main__.py",
    "forge/world_frame_decoder_adapt_v1/contract.py",
    "forge/world_frame_decoder_adapt_v1/runtime.py",
    "forge/world_frame_decoder_adapt_v1/training.py",
    "forge/world_frame_vae/model.py",
)


@dataclass(frozen=True, slots=True)
class TrainingPlan:
    total_updates: int = 1200
    segment_updates: int = 300
    batch_size: int = 4
    learning_rate: float = 4e-5
    ema_decay: float = 0.999
    original_probability: float = 0.25
    seed: int = 0x4445434F44455231

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
    digest = hashlib.sha256(b"nullvector-world-frame-decoder-adapt-v1\0")
    for relative in SOURCE_FILES:
        digest.update(relative.encode() + b"\0" + (PROJECT_ROOT / relative).read_bytes() + b"\0")
    return digest.hexdigest()
