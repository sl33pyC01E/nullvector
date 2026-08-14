from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Final

from ..config import PROJECT_ROOT


FORMAT: Final[str] = "nullvector-cellular-trauma-bank-v2"
ARRAY_FORMAT: Final[str] = "nullvector-cellular-trauma-arrays-v2"
DEFAULT_SOURCE: Final[Path] = PROJECT_ROOT / "outputs/cellular_physiology_v2/cellular_physiology_manifest.json"
DEFAULT_OUTPUT: Final[Path] = PROJECT_ROOT / "outputs/cellular_trauma_v2"
SCHEMA_PATH: Final[Path] = PROJECT_ROOT / "shared/schema/cellular_trauma_bank.schema.json"

HEAL_CLASS_NAMES: Final[tuple[str, ...]] = (
    "inert", "integument", "contractile", "vascular", "visceral", "neural", "stem_or_immune",
)

# Values are deliberately conservative.  A "polyp" is a detached, locally
# surviving component, not a claim that it has become a complete organism.
FAMILY_PROFILES: Final[tuple[dict[str, object], ...]] = (
    {"family": "humanoid", "reconnect_window_seconds": 4.0, "magnetic_radius_cells": 2.6, "magnetic_strength": 0.34, "clot_rate": 0.92, "scar_rate": 0.72, "regrowth_rate": 0.18, "detached_fate": "biomass", "polyp_min_cells": 9999},
    {"family": "animalian", "reconnect_window_seconds": 3.2, "magnetic_radius_cells": 2.4, "magnetic_strength": 0.30, "clot_rate": 0.80, "scar_rate": 0.60, "regrowth_rate": 0.24, "detached_fate": "biomass", "polyp_min_cells": 9999},
    {"family": "plantlike", "reconnect_window_seconds": 15.0, "magnetic_radius_cells": 3.4, "magnetic_strength": 0.24, "clot_rate": 0.54, "scar_rate": 0.36, "regrowth_rate": 0.88, "detached_fate": "polyp", "polyp_min_cells": 4},
    {"family": "anomaly", "reconnect_window_seconds": 24.0, "magnetic_radius_cells": 4.2, "magnetic_strength": 0.20, "clot_rate": 0.68, "scar_rate": 0.22, "regrowth_rate": 0.96, "detached_fate": "phase_polyp", "polyp_min_cells": 3},
    {"family": "machine", "reconnect_window_seconds": 18.0, "magnetic_radius_cells": 3.8, "magnetic_strength": 0.42, "clot_rate": 0.46, "scar_rate": 0.84, "regrowth_rate": 0.70, "detached_fate": "module_polyp", "polyp_min_cells": 5},
)

SOURCE_FILES: Final[tuple[str, ...]] = (
    "forge/cellular_trauma/__init__.py",
    "forge/cellular_trauma/__main__.py",
    "forge/cellular_trauma/contract.py",
    "forge/cellular_trauma/compiler.py",
    "forge/cellular_trauma/simulation.py",
    "shared/schema/cellular_trauma_bank.schema.json",
)


def source_sha256() -> str:
    digest = hashlib.sha256(b"nullvector-cellular-trauma-source-v2\0")
    for relative in SOURCE_FILES:
        path = PROJECT_ROOT / relative
        if not path.is_file(): raise FileNotFoundError(relative)
        digest.update(relative.encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()
