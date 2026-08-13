from __future__ import annotations

import os
from pathlib import Path
import shutil
from typing import Any, Mapping

from ..multifield_style.hashing import canonical_json_bytes, sha256_bytes, sha256_file


DISK_FLOOR_BYTES = 100 * 1024**3


def require_disk_floor(path: Path, *, planned_bytes: int = 0) -> dict[str, Any]:
    usage = shutil.disk_usage(Path(path).resolve().anchor)
    safe = usage.free - int(planned_bytes) >= DISK_FLOOR_BYTES
    result = {
        "free_bytes": int(usage.free),
        "floor_bytes": DISK_FLOOR_BYTES,
        "planned_bytes": int(planned_bytes),
        "safe": safe,
    }
    if not safe:
        raise OSError("Sprite-quality audit would cross the 100 GiB free-space floor")
    return result


def prepare_immutable_destination(path: Path) -> Path:
    destination = Path(path).resolve()
    require_disk_floor(destination.parent, planned_bytes=16 * 1024 * 1024)
    if destination.exists():
        if not destination.is_dir() or any(destination.iterdir()):
            raise FileExistsError(f"Sprite-quality destination must be empty: {destination}")
    else:
        destination.mkdir(parents=True, exist_ok=False)
    return destination


def write_bytes_new(path: Path, payload: bytes) -> None:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite sprite-quality artifact: {target}")
    require_disk_floor(target.parent, planned_bytes=max(len(payload) * 2, 256 * 1024))
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    write_bytes_new(path, canonical_json_bytes(payload))


def artifact_record(path: Path, root: Path) -> dict[str, Any]:
    target = Path(path).resolve()
    return {
        "path": target.relative_to(Path(root).resolve()).as_posix(),
        "bytes": target.stat().st_size,
        "sha256": sha256_file(target),
    }


def bytes_record(relative_path: str, payload: bytes) -> dict[str, Any]:
    return {
        "path": relative_path,
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
    }
