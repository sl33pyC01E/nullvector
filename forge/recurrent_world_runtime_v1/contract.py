from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..config import PROJECT_ROOT


FORMAT="nullvector-recurrent-world-runtime-v1/1.0.0"
V3_ROOT=PROJECT_ROOT/"outputs/recurrent_world_student_v3/production_v1"
V3_CHECKPOINT=V3_ROOT/"runtime.pt"
V3_REPORT=V3_ROOT/"report.json"
V3_SHA256="6b6123ca21a1115819ea71bb082f95b14983592b14aecbdf7cf07c0819472411"
ROLLOUT_REPORT=PROJECT_ROOT/"outputs/recurrent_world_rollout_v1/evaluation_v1/report.json"
CODEC_CHECKPOINT=PROJECT_ROOT/"outputs/world_frame_decoder_adapt_v1/production_v1/runtime.pt"
CODEC_SHA256="8b29795559876ce9e067e3b7a1addd72911d0704af43612afd887e227f86947a"
DEFAULT_OUTPUT=PROJECT_ROOT/"outputs/recurrent_world_runtime_v1/benchmark_v1"
SOURCE_FILES=(
    "forge/recurrent_world_runtime_v1/__init__.py",
    "forge/recurrent_world_runtime_v1/__main__.py",
    "forge/recurrent_world_runtime_v1/contract.py",
    "forge/recurrent_world_runtime_v1/runtime.py",
)


def canonical(value)->bytes:
    return (json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()


def file_sha256(path:Path)->str:
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1<<20),b""):digest.update(chunk)
    return digest.hexdigest()


def source_sha256()->str:
    digest=hashlib.sha256(b"nullvector-recurrent-world-runtime-v1\0")
    for relative in SOURCE_FILES:digest.update(relative.encode()+b"\0"+(PROJECT_ROOT/relative).read_bytes()+b"\0")
    return digest.hexdigest()
