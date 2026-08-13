from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .constants import PROJECT_ROOT


SOURCE_FILES = (
    "forge/neural_rig_repair/__init__.py",
    "forge/neural_rig_repair/__main__.py",
    "forge/neural_rig_repair/binding.py",
    "forge/neural_rig_repair/cli.py",
    "forge/neural_rig_repair/compiler.py",
    "forge/neural_rig_repair/constants.py",
    "forge/neural_rig_repair/hashing.py",
    "forge/neural_rig_repair/model.py",
    "forge/neural_rig_repair/motion.py",
    "forge/neural_rig_repair/planner.py",
    "forge/neural_rig_repair/replay.py",
    "forge/neural_rig_repair/schema.py",
    "forge/neural_rig_repair/source.py",
    "forge/neural_rig_repair/stress.py",
    "shared/schema/neural_rig_repair_plan.schema.json",
    "shared/schema/neural_rig_repair_bank.schema.json",
    "shared/schema/neural_rig_repair_replay.schema.json",
)


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, *, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(label: str, values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(b"nullvector-neural-rig-repair-array-v1\0")
    digest.update(label.encode("ascii"))
    digest.update(b"\0")
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(str(array.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes())
    return digest.hexdigest()


def source_hash() -> str:
    digest = hashlib.sha256()
    digest.update(b"nullvector-neural-rig-repair-source-v1\0")
    for relative in SOURCE_FILES:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Repair source file is missing: {relative}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def artifact_record(path: Path, root: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    root = Path(root).resolve()
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def write_canonical_json(path: Path, value: Any, *, replace: bool = False) -> None:
    path = Path(path)
    payload = canonical_json_bytes(value)
    if path.exists():
        if path.is_file() and not path.is_symlink() and path.read_bytes() == payload:
            return
        if not replace:
            raise FileExistsError(f"Refusing to overwrite non-identical artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"Atomic JSON staging path already exists: {temporary}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
