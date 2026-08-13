from __future__ import annotations

import os
from pathlib import Path
import shutil
from typing import Any, Mapping

import numpy as np
from PIL import Image

from .hashing import canonical_json_bytes


DISK_FLOOR_BYTES = 100 * 1024**3


def require_disk_floor(path: Path, *, planned_bytes: int = 0) -> dict[str, Any]:
    usage = shutil.disk_usage(Path(path).resolve().anchor)
    safe = usage.free - int(planned_bytes) >= DISK_FLOOR_BYTES
    status = {
        "total_bytes": int(usage.total),
        "used_bytes": int(usage.used),
        "free_bytes": int(usage.free),
        "floor_bytes": DISK_FLOOR_BYTES,
        "planned_bytes": int(planned_bytes),
        "safe": safe,
    }
    if not safe:
        raise OSError("Style compilation would cross the 100 GiB free-space floor")
    return status


def prepare_immutable_destination(destination: Path, *, planned_bytes: int) -> Path:
    destination = Path(destination).resolve()
    require_disk_floor(destination.parent, planned_bytes=planned_bytes)
    if destination.exists():
        if not destination.is_dir():
            raise FileExistsError(f"Style destination already exists as a file: {destination}")
        if any(destination.iterdir()):
            raise FileExistsError(
                f"Style destination is non-empty; derived banks are immutable: {destination}"
            )
    else:
        destination.mkdir(parents=True, exist_ok=False)
    return destination


def _temporary_path(path: Path, suffix: str) -> Path:
    return path.with_name(f".{path.stem}.{os.getpid()}.tmp{suffix}")


def write_bytes_new(path: Path, payload: bytes) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable style artifact: {path}")
    require_disk_floor(path.parent, planned_bytes=max(len(payload) * 2, 256 * 1024))
    temporary = _temporary_path(path, ".bin")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    write_bytes_new(path, canonical_json_bytes(payload))


def write_png_new(path: Path, pixels: np.ndarray) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable style artifact: {path}")
    values = np.asarray(pixels)
    if values.dtype != np.uint8 or values.ndim != 3 or values.shape[2] not in (3, 4):
        raise ValueError("PNG pixels must be uint8 RGB or RGBA")
    require_disk_floor(path.parent, planned_bytes=max(int(values.nbytes * 4), 256 * 1024))
    temporary = _temporary_path(path, ".png")
    try:
        Image.fromarray(values).save(
            temporary,
            format="PNG",
            optimize=False,
            compress_level=9,
        )
        if temporary.stat().st_size <= 0:
            raise RuntimeError("Temporary PNG is empty")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
