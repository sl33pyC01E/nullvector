from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..config import PROJECT_ROOT


FORMAT = "nullvector-contextual-recurrent-world-pipeline-v1/1.0.0"
DEFAULT_RELEASE = PROJECT_ROOT / "outputs/contextual_recurrent_world_pipeline_v1/release.json"
WORLD_STATE = PROJECT_ROOT / "examples/models/neural_world_state_v1.pt"
WORLD_STATE_SHA256 = "98956a07db713306ab2e772943d95b5999cf94b57062b65b89bc275066e64184"
CONTEXT_ADAPTER = PROJECT_ROOT / "examples/models/recurrent_world_context_v1.pt"
CONTEXT_ADAPTER_SHA256 = "78c540fcaf402a5db0530cad35acb7f602066f1c1b15d5316d25dd818e869763"
SOURCE_FILES = (
    "forge/contextual_recurrent_world_pipeline_v1/__init__.py",
    "forge/contextual_recurrent_world_pipeline_v1/__main__.py",
    "forge/contextual_recurrent_world_pipeline_v1/contract.py",
    "forge/contextual_recurrent_world_pipeline_v1/runtime.py",
    "forge/contextual_recurrent_world_pipeline_v1/release.py",
)


def canonical(value: object) -> bytes: return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""): digest.update(chunk)
    return digest.hexdigest()


def source_sha256() -> str:
    digest = hashlib.sha256(b"nullvector-contextual-recurrent-world-pipeline-v1\0")
    for relative in SOURCE_FILES: digest.update(relative.encode() + b"\0" + (PROJECT_ROOT / relative).read_bytes() + b"\0")
    return digest.hexdigest()
