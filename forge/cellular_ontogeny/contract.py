from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Final

from ..config import PROJECT_ROOT


FORMAT: Final[str] = "nullvector-cellular-ontogeny-bank-v1"
PROGRAM_FORMAT: Final[str] = "nullvector-cellular-ontogeny-program-v1"
ARRAY_FORMAT: Final[str] = "nullvector-cellular-ontogeny-arrays-v1"
DEFAULT_SOURCE: Final[Path] = PROJECT_ROOT / "outputs/cellular_breeding_symmetry_v1/cellular_symmetry_manifest.json"
DEFAULT_OUTPUT: Final[Path] = PROJECT_ROOT / "outputs/cellular_ontogeny_v1"
SCHEMA_PATH: Final[Path] = PROJECT_ROOT / "shared/schema/cellular_ontogeny_bank.schema.json"
STAGES: Final[tuple[str, ...]] = ("zygote", "gastrula", "organ_primordia", "larval", "juvenile", "adult")
STAGE_FRACTIONS: Final[tuple[float, ...]] = (0.012, 0.08, 0.25, 0.50, 0.75, 1.0)
ARRAY_NAMES: Final[tuple[str, ...]] = (
    "birth_order", "activation_stage", "parent_cell", "lineage_id", "differentiation_time",
    "bond_activation_stage", "morphogen_lr", "morphogen_ap", "morphogen_core",
)
SOURCE_FILES: Final[tuple[str, ...]] = (
    "forge/cellular_ontogeny/__init__.py", "forge/cellular_ontogeny/__main__.py",
    "forge/cellular_ontogeny/contract.py", "forge/cellular_ontogeny/compiler.py",
    "shared/schema/cellular_ontogeny_bank.schema.json",
)


def source_sha256() -> str:
    digest = hashlib.sha256(b"nullvector-cellular-ontogeny-source-v1\0")
    for relative in SOURCE_FILES:
        path = PROJECT_ROOT / relative
        if not path.is_file(): raise FileNotFoundError(relative)
        digest.update(relative.encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()
