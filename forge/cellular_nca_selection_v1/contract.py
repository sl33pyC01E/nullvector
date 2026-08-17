from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..config import PROJECT_ROOT


FORMAT = "nullvector-causal-cellular-nca-selection-v1"
RUNTIME_FORMAT = "nullvector-causal-cellular-nca-runtime-v1"
DEFAULT_AUTHORITY = PROJECT_ROOT / "outputs/cellular_nca/nca_causal_v3_long_horizon"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/cellular_nca/nca_causal_v3_selected"
SOURCE_FILES = (
    "forge/cellular_nca_selection_v1/__init__.py",
    "forge/cellular_nca_selection_v1/__main__.py",
    "forge/cellular_nca_selection_v1/contract.py",
    "forge/cellular_nca_selection_v1/evaluation.py",
)


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_sha256() -> str:
    digest = hashlib.sha256(b"nullvector-causal-cellular-nca-selection-v1\0")
    for relative in SOURCE_FILES:
        digest.update(relative.encode() + b"\0" + (PROJECT_ROOT / relative).read_bytes() + b"\0")
    digest.update(sha256_file(DEFAULT_AUTHORITY / "causal_v3_segment_0000256.pt").encode())
    return digest.hexdigest()
