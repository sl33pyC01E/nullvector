from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Final

from ..config import PROJECT_ROOT


FORMAT: Final[str] = "nullvector-cellular-ecology-bank-v1"
MAP_FORMAT: Final[str] = "nullvector-cellular-ecology-map-v1"
FIELD_FORMAT: Final[str] = "nullvector-cellular-ecology-fields-v1"
DEFAULT_MAP_ROOT: Final[Path] = PROJECT_ROOT / "outputs/maps_v2_forge_lab"
DEFAULT_ORGANISM_SOURCE: Final[Path] = PROJECT_ROOT / "outputs/cellular_breeding_symmetry_v1/cellular_symmetry_manifest.json"
DEFAULT_OUTPUT: Final[Path] = PROJECT_ROOT / "outputs/cellular_ecology_v1"
SCHEMA_PATH: Final[Path] = PROJECT_ROOT / "shared/schema/cellular_ecology_bank.schema.json"
FAMILIES: Final[tuple[str, ...]] = ("humanoid", "animalian", "plantlike", "anomaly", "machine")
FIELD_NAMES: Final[tuple[str, ...]] = (
    "nutrient", "moisture", "light", "temperature", "toxicity", "energy", "biomass",
)
RESOURCE_NAMES: Final[tuple[str, ...]] = (
    "none", "organic_food", "water", "photonic_bloom", "anomalous_plasma", "mineral_charge",
)

SOURCE_FILES: Final[tuple[str, ...]] = (
    "forge/cellular_ecology/__init__.py",
    "forge/cellular_ecology/__main__.py",
    "forge/cellular_ecology/contract.py",
    "forge/cellular_ecology/compiler.py",
    "shared/schema/cellular_ecology_bank.schema.json",
)


def source_sha256() -> str:
    digest = hashlib.sha256(b"nullvector-cellular-ecology-source-v1\0")
    for relative in SOURCE_FILES:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()
