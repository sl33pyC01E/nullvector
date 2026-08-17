from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import torch

from ..config import PROJECT_ROOT

FORMAT = "nullvector-world-frame-rollout-decoder-v3/3.0.0"
CHECKPOINT_FORMAT = FORMAT + "-checkpoint"
REPORT_FORMAT = FORMAT + "-report"
ROLLOUT_CORPUS = PROJECT_ROOT / "outputs/world_frame_rollout_decoder_v2/corpus_v1"
ROLLOUT_CORPUS_SHA256 = "97ec406ed32cb3bcb755e98b565af07b05129c1c511f4adc7ff06532788616cf"
PARENT = PROJECT_ROOT / "outputs/world_frame_rollout_decoder_v2/production_v1/runtime.pt"
PARENT_SHA256 = "68cf898b54948091b225156fa5c357072f81322c0001a4335365c1efb4ca01e2"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/world_frame_rollout_decoder_v3/production_v1"
SOURCE_FILES = (
    "forge/world_frame_rollout_decoder_v3/__init__.py",
    "forge/world_frame_rollout_decoder_v3/__main__.py",
    "forge/world_frame_rollout_decoder_v3/contract.py",
    "forge/world_frame_rollout_decoder_v3/training.py",
    "forge/world_frame_vae/model.py",
)


@dataclass(frozen=True, slots=True)
class TrainingPlan:
    total_updates: int = 1200
    segment_updates: int = 300
    batch_size: int = 4
    learning_rate: float = 2e-5
    ema_decay: float = 0.999
    authoritative_probability: float = 0.30
    foreground_weight: float = 10.0
    seed: int = 0x464F43414C444543

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
    digest = hashlib.sha256(b"nullvector-world-frame-rollout-decoder-v3\0")
    for relative in SOURCE_FILES:
        digest.update(relative.encode() + b"\0" + (PROJECT_ROOT / relative).read_bytes() + b"\0")
    return digest.hexdigest()
