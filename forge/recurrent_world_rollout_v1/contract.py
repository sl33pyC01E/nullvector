from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..config import PROJECT_ROOT

FORMAT = "nullvector-recurrent-world-rollout-v1/1.0.0"
V3_CHECKPOINT = PROJECT_ROOT / "outputs/recurrent_world_student_v3/production_v1/runtime.pt"
V3_SHA256 = "6b6123ca21a1115819ea71bb082f95b14983592b14aecbdf7cf07c0819472411"
ACTION_CHECKPOINT = PROJECT_ROOT / "outputs/recurrent_action_dit_v2/production_v1/runtime.pt"
ACTOR_CHECKPOINT = PROJECT_ROOT / "outputs/actor_state_student_v1/production_v1/actor_0000800.pt"
CODEC_CHECKPOINT = PROJECT_ROOT / "outputs/world_frame_decoder_adapt_v1/production_v1/runtime.pt"
CORPUS = PROJECT_ROOT / "outputs/world_action_contiguous_v8/corpus_v1_6world"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/recurrent_world_rollout_v1/evaluation_v1"
SOURCE_FILES = (
    "forge/recurrent_world_rollout_v1/__init__.py",
    "forge/recurrent_world_rollout_v1/__main__.py",
    "forge/recurrent_world_rollout_v1/contract.py",
    "forge/recurrent_world_rollout_v1/evaluation.py",
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
    digest = hashlib.sha256(b"nullvector-recurrent-world-rollout-v1\0")
    for relative in SOURCE_FILES:
        digest.update(relative.encode() + b"\0" + (PROJECT_ROOT / relative).read_bytes() + b"\0")
    return digest.hexdigest()
