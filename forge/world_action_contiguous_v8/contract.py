from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..config import PROJECT_ROOT

FORMAT = "nullvector-contiguous-cellular-action-corpus/8.0.0"
VAE_CHECKPOINT = PROJECT_ROOT / "outputs/world_frame_vae/production_v2_high_fidelity/checkpoint.pt"
VAE_CHECKPOINT_SHA256 = "875691e4be9866000ea4a112ca708ccd0755fa98d3fdededa3bb09bf3b560259"
SOURCE_ROOT = PROJECT_ROOT / "outputs/action_teacher_v2_production_v1"
SOURCE_NAMES = tuple(f"cellular-v7-world-{letter}" for letter in "abcdef")
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/world_action_contiguous_v8/corpus_v1_6world"
SOURCE_FILES = (
    "forge/world_action_contiguous_v8/__init__.py",
    "forge/world_action_contiguous_v8/__main__.py",
    "forge/world_action_contiguous_v8/contract.py",
    "forge/world_action_contiguous_v8/corpus.py",
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
    digest = hashlib.sha256(b"nullvector-contiguous-cellular-action-corpus-v8\0")
    for relative in SOURCE_FILES:
        digest.update(relative.encode() + b"\0" + (PROJECT_ROOT / relative).read_bytes() + b"\0")
    return digest.hexdigest()
