from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import torch

from ..config import PROJECT_ROOT

FORMAT = "nullvector-world-frame-rollout-decoder-v2/2.0.0"
CORPUS_FORMAT = FORMAT + "-corpus"
CHECKPOINT_FORMAT = FORMAT + "-checkpoint"
REPORT_FORMAT = FORMAT + "-report"
NATURAL_CORPUS = PROJECT_ROOT / "outputs/world_action_natural_v10/corpus_v1_6world"
NATURAL_CORPUS_SHA256 = "e96b10f80db3e824fdb768dc9e52ac8ff5e7f228cf3b87ba89d1df8d3047662f"
RECURRENT = PROJECT_ROOT / "outputs/recurrent_world_student_v6/production_v1/runtime_calibrated_ramp.pt"
RECURRENT_SHA256 = "1516633d413aa19930dea53d0eb5a526d8528761e4120f4a0e9b70da42489b64"
PARENT_CODEC = PROJECT_ROOT / "outputs/world_frame_decoder_adapt_v1/production_v1/runtime.pt"
PARENT_CODEC_SHA256 = "8b29795559876ce9e067e3b7a1addd72911d0704af43612afd887e227f86947a"
DEFAULT_CORPUS = PROJECT_ROOT / "outputs/world_frame_rollout_decoder_v2/corpus_v1"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/world_frame_rollout_decoder_v2/production_v1"
HORIZONS = (1, 2, 4, 8, 16, 32)
SOURCE_FILES = (
    "forge/world_frame_rollout_decoder_v2/__init__.py",
    "forge/world_frame_rollout_decoder_v2/__main__.py",
    "forge/world_frame_rollout_decoder_v2/contract.py",
    "forge/world_frame_rollout_decoder_v2/corpus.py",
    "forge/world_frame_rollout_decoder_v2/training.py",
    "forge/world_frame_vae/model.py",
)


@dataclass(frozen=True, slots=True)
class TrainingPlan:
    total_updates: int = 1200
    segment_updates: int = 300
    batch_size: int = 4
    learning_rate: float = 3e-5
    ema_decay: float = 0.999
    authoritative_probability: float = 0.30
    seed: int = 0x524F4C4C44454332

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
    digest = hashlib.sha256(b"nullvector-world-frame-rollout-decoder-v2\0")
    for relative in SOURCE_FILES:
        digest.update(relative.encode() + b"\0" + (PROJECT_ROOT / relative).read_bytes() + b"\0")
    return digest.hexdigest()
