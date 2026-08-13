from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Final

from ..cellular_organism.contract import CELLULAR_CONTRACT_SHA256
from ..config import PROJECT_ROOT
from ..evolved_cellular_organism.contract import source_sha256 as evolved_cellular_source_sha256


FORMAT: Final[str] = "nullvector-cellular-breeding-bank-v1"
SPECIES_FORMAT: Final[str] = "nullvector-cellular-breeding-offspring-v1"
FIELDS_FORMAT: Final[str] = "nullvector-cellular-breeding-fields-v1"
DEFAULT_SOURCE: Final[Path] = PROJECT_ROOT / "outputs/evolved_cellular_organism_v1/evolved_cellular_organism_manifest.json"
DEFAULT_OUTPUT: Final[Path] = PROJECT_ROOT / "outputs/cellular_breeding_v1"
SCHEMA_PATH: Final[Path] = PROJECT_ROOT / "shared/schema/cellular_breeding_bank.schema.json"
OFFSPRING_COUNT: Final[int] = 45

CROSSOVER_MODES: Final[tuple[str, ...]] = (
    "sagittal_splice",
    "transverse_splice",
    "radial_graft",
    "voronoi_weave",
    "organ_graft",
    "cellular_mosaic",
)
MUTATION_MODES: Final[tuple[str, ...]] = (
    "none",
    "budding_growth",
    "boundary_apoptosis",
    "armor_metaplasia",
    "bioluminescent_shift",
    "appendage_graft",
)

SOURCE_FILES: Final[tuple[str, ...]] = (
    "forge/cellular_breeding/__init__.py",
    "forge/cellular_breeding/__main__.py",
    "forge/cellular_breeding/contract.py",
    "forge/cellular_breeding/compiler.py",
    "forge/cellular_organism/compiler.py",
    "shared/schema/cellular_breeding_bank.schema.json",
)


def source_sha256() -> str:
    digest = hashlib.sha256(b"nullvector-cellular-breeding-source-v1\0")
    for dependency in (CELLULAR_CONTRACT_SHA256, evolved_cellular_source_sha256()):
        digest.update(dependency.encode("ascii")); digest.update(b"\0")
    for relative in SOURCE_FILES:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        digest.update(relative.encode("utf-8")); digest.update(b"\0")
        digest.update(path.read_bytes()); digest.update(b"\0")
    return digest.hexdigest()
