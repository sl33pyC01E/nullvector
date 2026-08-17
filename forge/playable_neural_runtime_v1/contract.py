from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..config import PROJECT_ROOT

FORMAT = "nullvector-playable-neural-runtime-v1/1.0.0"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/playable_neural_runtime_v1/build_001"
ENSEMBLE = PROJECT_ROOT / "outputs/neural_ensemble_v1/build_001/ensemble_manifest.json"
COMPOSITE = PROJECT_ROOT / "outputs/composite_world_v1/build_002/composite_manifest.json"
SOURCE_FILES = (
    "forge/playable_neural_runtime_v1/__init__.py",
    "forge/playable_neural_runtime_v1/__main__.py",
    "forge/playable_neural_runtime_v1/contract.py",
    "forge/playable_neural_runtime_v1/runtime.py",
    "forge/playable_neural_runtime_v1/release.py",
)


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_sha256() -> str:
    digest = hashlib.sha256(b"nullvector-playable-neural-runtime-v1\0")
    for relative in SOURCE_FILES:
        digest.update(relative.encode() + b"\0" + (PROJECT_ROOT / relative).read_bytes() + b"\0")
    return digest.hexdigest()
