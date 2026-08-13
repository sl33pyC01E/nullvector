from __future__ import annotations

import hashlib
from pathlib import Path

from ..config import PROJECT_ROOT


SOURCE_FILES = (
    "forge/map_art/__init__.py",
    "forge/map_art/atlas.py",
    "forge/map_art/autotile.py",
    "forge/map_art/hashing.py",
    "forge/map_art/io.py",
    "forge/map_art/model.py",
    "forge/map_art/objects.py",
    "forge/map_art/provenance.py",
    "forge/map_art/renderer.py",
    "forge/map_art/styles.py",
    "forge/map_art/validate.py",
    "forge/maps/model.py",
    "shared/schema/map_art_manifest.schema.json",
)


def source_hash(root: Path = PROJECT_ROOT) -> str:
    digest = hashlib.sha256()
    root = Path(root)
    for relative in SOURCE_FILES:
        path = root / relative
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()

