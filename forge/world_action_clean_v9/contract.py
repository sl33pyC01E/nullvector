from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..config import PROJECT_ROOT


FORMAT = "nullvector-clean-contiguous-world-action-corpus/9.0.0"
CODEC_CHECKPOINT = PROJECT_ROOT / "outputs/world_frame_decoder_adapt_v1/production_v1/runtime.pt"
CODEC_SHA256 = "8b29795559876ce9e067e3b7a1addd72911d0704af43612afd887e227f86947a"
SOURCE_ROOT = PROJECT_ROOT / "outputs/action_teacher_clean_v3/production_v2"
SOURCE_NAMES = tuple(f"clean-world-{letter}" for letter in "abcdef")
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/world_action_clean_v9/corpus_v1_6world"
SOURCE_FILES = (
    "forge/world_action_clean_v9/__init__.py",
    "forge/world_action_clean_v9/__main__.py",
    "forge/world_action_clean_v9/contract.py",
    "forge/world_action_clean_v9/corpus.py",
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
    digest = hashlib.sha256(b"nullvector-clean-contiguous-world-action-corpus-v9\0")
    for relative in SOURCE_FILES:
        digest.update(relative.encode() + b"\0" + (PROJECT_ROOT / relative).read_bytes() + b"\0")
    return digest.hexdigest()
