from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import numpy as np


MASK64 = (1 << 64) - 1


def mix64(value: int) -> int:
    """SplitMix64 finalizer with explicit unsigned wrapping."""
    value = (int(value) + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return (value ^ (value >> 31)) & MASK64


def coordinate_hash(seed: int, x: int, y: int, salt: int = 0) -> int:
    """Random-access deterministic hash for one integer grid coordinate."""
    value = int(seed) & MASK64
    value ^= (int(x) * 0xD6E8FEB86659FD93) & MASK64
    value ^= (int(y) * 0xA5A3564E27F8862D) & MASK64
    value ^= (int(salt) * 0x9E3779B97F4A7C15) & MASK64
    return mix64(value)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii") + b"\0")
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def named_arrays_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    """Hash named arrays without depending on archive or dictionary ordering."""
    digest = hashlib.sha256()
    for name in sorted(arrays):
        contiguous = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(contiguous.dtype.str.encode("ascii") + b"\0")
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()
