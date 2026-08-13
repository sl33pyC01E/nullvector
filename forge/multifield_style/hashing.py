from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SHA256_HEX_LENGTH = 64


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, *, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def aligned_fields_hash(
    part: np.ndarray,
    material: np.ndarray,
    emission: np.ndarray,
) -> str:
    """Hash categorical fields using the evaluator's published byte contract.

    This implementation intentionally lives in the presentation package so the
    compiler does not import the training/evaluation runtime (and therefore
    never initializes torch or CUDA).
    """

    digest = hashlib.sha256()
    digest.update(b"nullvector-aligned-fields-v1\0")
    for name, values in (
        ("part", part),
        ("material", material),
        ("emission", emission),
    ):
        array = np.ascontiguousarray(values)
        digest.update(name.encode("ascii"))
        digest.update(b"\0")
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(str(array.shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(array.tobytes())
    return digest.hexdigest()


def artifact_record(path: Path, root: Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    root_resolved = Path(root).resolve()
    relative = resolved.relative_to(root_resolved).as_posix()
    return {
        "path": relative,
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == SHA256_HEX_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )
