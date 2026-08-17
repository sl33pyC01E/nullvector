from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..config import PROJECT_ROOT

FORMAT = "nullvector-neural-action-frame-v1/1.0.0"
REPORT_FORMAT = FORMAT + "-report"
ACTION_OUTPUT = PROJECT_ROOT / "outputs/recurrent_action_dit_v2/production_v1"
ACTION_CHECKPOINT_SHA256 = "22c3c9a23a411057bf376bee77b750318f459414f35bd4d87ab6e326b90323ef"
CODEC_CHECKPOINT = PROJECT_ROOT / "outputs/world_frame_decoder_adapt_v1/production_v1/runtime.pt"
CODEC_CHECKPOINT_SHA256 = "8b29795559876ce9e067e3b7a1addd72911d0704af43612afd887e227f86947a"
CORPUS = PROJECT_ROOT / "outputs/world_action_cellular_v7/corpus_v1_6world"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/neural_action_frame_v1/evaluation_v1"
SOURCE_FILES = (
    "forge/neural_action_frame_v1/__init__.py",
    "forge/neural_action_frame_v1/__main__.py",
    "forge/neural_action_frame_v1/contract.py",
    "forge/neural_action_frame_v1/evaluation.py",
    "forge/neural_action_frame_v1/runtime.py",
)


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def file_sha256(path: Path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_sha256():
    digest = hashlib.sha256(b"nullvector-neural-action-frame-v1\0")
    for relative in SOURCE_FILES:
        digest.update(relative.encode() + b"\0" + (PROJECT_ROOT / relative).read_bytes() + b"\0")
    return digest.hexdigest()
