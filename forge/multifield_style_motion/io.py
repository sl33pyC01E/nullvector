from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
import shutil
from typing import Any, Mapping

from .hashing import canonical_json_bytes, sha256_bytes, sha256_file


DISK_FLOOR_BYTES = 100 * 1024**3


def require_disk_floor(path: Path, *, planned_bytes: int = 0) -> dict[str, Any]:
    root = Path(path).resolve().anchor
    usage = shutil.disk_usage(root)
    safe = usage.free - int(planned_bytes) >= DISK_FLOOR_BYTES
    status = {
        "free_bytes": int(usage.free),
        "planned_bytes": int(planned_bytes),
        "floor_bytes": DISK_FLOOR_BYTES,
        "safe": safe,
    }
    if not safe:
        raise OSError("Style-motion compilation would cross the 100 GiB free-space floor")
    return status


def write_exact(path: Path, payload: bytes) -> None:
    """Create atomically, or accept a byte-exact existing resume artifact."""

    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.is_symlink() or path.read_bytes() != payload:
            raise FileExistsError(f"Existing style-motion artifact is not byte-exact: {path}")
        return
    require_disk_floor(path.parent, planned_bytes=max(len(payload) * 2, 256 * 1024))
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_json_exact(path: Path, payload: Mapping[str, Any]) -> None:
    write_exact(path, canonical_json_bytes(payload))


def safe_resolve(root: Path, relative: object) -> Path:
    if not isinstance(relative, str):
        raise ValueError("Artifact path must be a string")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"Unsafe artifact path: {relative!r}")
    if "\\" in relative:
        raise ValueError("Artifact paths must use POSIX separators")
    target = (Path(root).resolve() / Path(*pure.parts)).resolve()
    try:
        target.relative_to(Path(root).resolve())
    except ValueError as error:
        raise ValueError(f"Artifact escapes its root: {relative}") from error
    if not target.is_file() or target.is_symlink():
        raise ValueError(f"Artifact must be a regular non-symlink file: {relative}")
    return target


def verify_artifact(root: Path, record: Mapping[str, Any]) -> Path:
    path = safe_resolve(root, record.get("path"))
    expected_bytes = record.get("bytes")
    expected_hash = record.get("sha256")
    if path.stat().st_size != expected_bytes:
        raise ValueError(f"Artifact byte count mismatch: {record.get('path')}")
    if sha256_file(path) != expected_hash:
        raise ValueError(f"Artifact SHA-256 mismatch: {record.get('path')}")
    return path


def artifact_record(path: Path, root: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    root = Path(root).resolve()
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
