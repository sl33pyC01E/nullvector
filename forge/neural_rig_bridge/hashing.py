from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_hash(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def array_hash(label: str, values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(b"nullvector-array-v1\0")
    digest.update(label.encode("utf-8"))
    digest.update(b"\0")
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(str(array.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes())
    return digest.hexdigest()


def aligned_fields_hash(
    part: np.ndarray, material: np.ndarray, emission: np.ndarray
) -> str:
    """Match the authoritative multi-field evaluator's hash byte-for-byte."""
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


def owner_tuple_hash(
    owner_id: int,
    mask: np.ndarray,
    material: np.ndarray,
    emission: np.ndarray,
) -> str:
    points = np.argwhere(mask)
    digest = hashlib.sha256()
    digest.update(b"nullvector-owner-tuple-layer-v1\0")
    digest.update(int(owner_id).to_bytes(2, "little", signed=False))
    for y, x in points:
        digest.update(bytes((int(x), int(y), owner_id)))
        digest.update(bytes((int(material[y, x]), int(emission[y, x]))))
    return digest.hexdigest()


_SOURCE_FILES = (
    "__init__.py",
    "adapter.py",
    "binding.py",
    "hashing.py",
    "model.py",
    "motion_program.py",
    "replay.py",
    "validation.py",
)


def binder_source_hash(root: Path | None = None) -> str:
    package = Path(__file__).resolve().parent if root is None else Path(root).resolve()
    project = package.parent.parent
    digest = hashlib.sha256()
    digest.update(b"nullvector-neural-rig-bridge-source-v1\0")
    for name in _SOURCE_FILES:
        path = package / name
        relative = f"forge/neural_rig_bridge/{name}"
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    for relative in (
        "forge/morphology/constants.py",
        "forge/morphology/motion.py",
        "shared/schema/multifield_raw_sample.schema.json",
        "shared/schema/neural_rig_binding.schema.json",
        "shared/schema/neural_rig_motion.schema.json",
    ):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update((project / relative).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def tuple_fingerprint(tuples: np.ndarray | Iterable[tuple[int, int, int]]) -> str:
    values = np.asarray(list(tuples) if not isinstance(tuples, np.ndarray) else tuples)
    values = np.ascontiguousarray(values, dtype=np.uint8)
    return array_hash("legal_part_material_emission_tuples", values)


def evaluator_tuple_fingerprint(
    tuples: np.ndarray | Iterable[tuple[int, int, int]],
) -> str:
    """Match the evaluator/checkpoint legal-table fingerprint byte-for-byte."""
    values = np.asarray(list(tuples) if not isinstance(tuples, np.ndarray) else tuples)
    values = np.ascontiguousarray(values, dtype=np.uint8)
    if values.ndim != 2 or values.shape[1:] != (3,) or not len(values):
        raise ValueError("legal tuples must be a nonempty [N, 3] table")
    return hashlib.sha256(values.tobytes()).hexdigest()
