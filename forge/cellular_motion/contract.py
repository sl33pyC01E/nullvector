from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Final

from ..cellular_symmetry.contract import source_sha256 as symmetry_source_sha256
from ..config import PROJECT_ROOT


FORMAT: Final[str] = "nullvector-cellular-neuromuscular-motion-bank-v1"
DEFAULT_SOURCE: Final[Path] = PROJECT_ROOT / "outputs/cellular_breeding_symmetry_v1/cellular_symmetry_manifest.json"
DEFAULT_OUTPUT: Final[Path] = PROJECT_ROOT / "outputs/cellular_motion_v1"
SCHEMA_PATH: Final[Path] = PROJECT_ROOT / "shared/schema/cellular_motion_bank.schema.json"

FACING_NAMES: Final[tuple[str, ...]] = ("north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest")
MOTION_NAMES: Final[tuple[str, ...]] = (
    "idle_breathe", "idle_wiggle", "locomote", "joy", "anger", "fear", "confused",
    "sleep", "taunt", "attack", "cast", "hit", "death",
)
DRIVER_NAMES: Final[tuple[str, ...]] = (
    "body_bob", "body_sway", "body_squash", "head_tilt",
    "appendage_left", "appendage_right", "locomotor_left", "locomotor_right",
    "auxiliary", "weapon_recoil", "sensory_focus", "emission_pulse",
    "propulsion", "pain_spasm",
)
MOTION_SPECS: Final[dict[str, tuple[int, int, bool]]] = {
    "idle_breathe": (9, 8, True), "idle_wiggle": (9, 10, True), "locomote": (9, 12, True),
    "joy": (9, 10, True), "anger": (9, 12, True), "fear": (13, 14, True),
    "confused": (9, 9, True), "sleep": (9, 6, True), "taunt": (9, 11, True),
    "attack": (8, 14, False), "cast": (8, 12, False), "hit": (7, 16, False), "death": (10, 10, False),
}

SOURCE_FILES: Final[tuple[str, ...]] = (
    "forge/cellular_motion/__init__.py", "forge/cellular_motion/__main__.py",
    "forge/cellular_motion/contract.py", "forge/cellular_motion/compiler.py",
    "shared/schema/cellular_motion_bank.schema.json",
)


def source_sha256() -> str:
    digest = hashlib.sha256(b"nullvector-cellular-neuromuscular-motion-source-v1\0")
    digest.update(symmetry_source_sha256().encode("ascii")); digest.update(b"\0")
    for relative in SOURCE_FILES:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        digest.update(relative.encode("utf-8")); digest.update(b"\0"); digest.update(path.read_bytes()); digest.update(b"\0")
    return digest.hexdigest()
