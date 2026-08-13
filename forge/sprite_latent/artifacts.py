from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DISK_FLOOR_BYTES = 100 * 1024**3
SOURCE_FILES = (
    "forge/sprite_latent/__init__.py",
    "forge/sprite_latent/__main__.py",
    "forge/sprite_latent/artifacts.py",
    "forge/sprite_latent/codec.py",
    "forge/sprite_latent/corpus.py",
    "forge/sprite_latent/schema.py",
    "forge/sprite_latent/smoke.py",
    "forge/sprite_latent/training.py",
    "shared/schema/sprite_latent_smoke.schema.json",
)


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True)
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


def source_hash(project_root: Path = PROJECT_ROOT) -> str:
    root = Path(project_root).resolve()
    digest = hashlib.sha256()
    digest.update(b"nullvector-sprite-fsq-source-v1\0")
    for relative in SOURCE_FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Sprite latent source member missing: {relative}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


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
        raise OSError("Sprite latent work would cross the 100 GiB free-space floor")
    return result


def prepare_destination(path: Path) -> Path:
    destination = Path(path).resolve()
    require_disk_floor(destination.parent, planned_bytes=64 * 1024 * 1024)
    if destination.exists():
        if not destination.is_dir() or any(destination.iterdir()):
            raise FileExistsError(f"Sprite latent destination must be empty: {destination}")
    else:
        destination.mkdir(parents=True, exist_ok=False)
    return destination


def write_bytes_new(path: Path, payload: bytes) -> None:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite sprite latent artifact: {target}")
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
