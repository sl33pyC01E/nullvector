from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8") + b"\n"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def array_sha256(name: str, values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(name.encode("ascii"))
    digest.update(b"\0")
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical_json_bytes(list(array.shape)))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


FOUNDATION_SOURCE_FILES = (
    "__init__.py",
    "genetics.py",
    "hashing.py",
    "model.py",
    "rig.py",
)


def _source_hash(names: tuple[str, ...]) -> str:
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in names:
        path = root / name
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def source_hash() -> str:
    """Hash only the immutable field-fusion and fresh-rig foundation."""

    return _source_hash(FOUNDATION_SOURCE_FILES)


def pilot_source_hash() -> str:
    return _source_hash((*FOUNDATION_SOURCE_FILES, "pilot.py"))


def stress_source_hash() -> str:
    return _source_hash((*FOUNDATION_SOURCE_FILES, "stress.py"))
