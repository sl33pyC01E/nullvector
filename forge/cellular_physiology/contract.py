from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Final

from ..config import PROJECT_ROOT


FORMAT: Final[str] = "nullvector-connected-cellular-physiology-bank-v2"
ARRAY_FORMAT: Final[str] = "nullvector-connected-cellular-physiology-arrays-v2"
DEFAULT_SOURCE: Final[Path] = PROJECT_ROOT / "outputs/cellular_breeding_symmetry_v1/cellular_symmetry_manifest.json"
DEFAULT_OUTPUT: Final[Path] = PROJECT_ROOT / "outputs/cellular_physiology_v2"
SCHEMA_PATH: Final[Path] = PROJECT_ROOT / "shared/schema/cellular_physiology_bank.schema.json"

SYSTEM_NAMES: Final[tuple[str, ...]] = (
    "circulation", "respiration", "digestion", "neural",
    "sensory", "locomotion", "reproduction", "immune",
)
ROLE_NAMES: Final[tuple[str, ...]] = ("none", "core", "conduit", "exchange_or_effector")
DEPENDENCIES: Final[dict[str, tuple[str, ...]]] = {
    "circulation": (),
    "respiration": ("circulation",),
    "digestion": ("circulation",),
    "neural": ("circulation", "respiration"),
    "sensory": ("neural",),
    "locomotion": ("neural", "circulation", "respiration"),
    "reproduction": ("circulation", "respiration", "digestion"),
    "immune": ("circulation", "digestion"),
}

SOURCE_FILES: Final[tuple[str, ...]] = (
    "forge/cellular_physiology/__init__.py",
    "forge/cellular_physiology/__main__.py",
    "forge/cellular_physiology/contract.py",
    "forge/cellular_physiology/compiler.py",
    "forge/cellular_physiology/simulation.py",
    "shared/schema/cellular_physiology_bank.schema.json",
)


def source_sha256() -> str:
    digest = hashlib.sha256(b"nullvector-connected-cellular-physiology-source-v2\0")
    for relative in SOURCE_FILES:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        digest.update(relative.encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()
