from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Final

from ..cellular_breeding.contract import source_sha256 as breeding_source_sha256
from ..cellular_organism.contract import CELLULAR_CONTRACT_SHA256
from ..config import PROJECT_ROOT


FORMAT: Final[str] = "nullvector-cellular-symmetry-bank-v1"
SPECIES_FORMAT: Final[str] = "nullvector-cellular-symmetry-offspring-v1"
FIELDS_FORMAT: Final[str] = "nullvector-cellular-symmetry-fields-v1"
DEFAULT_SOURCE: Final[Path] = PROJECT_ROOT / "outputs/cellular_breeding_v1/cellular_breeding_manifest.json"
DEFAULT_OUTPUT: Final[Path] = PROJECT_ROOT / "outputs/cellular_breeding_symmetry_v1"
SCHEMA_PATH: Final[Path] = PROJECT_ROOT / "shared/schema/cellular_symmetry_bank.schema.json"
SAMPLE_COUNT: Final[int] = 45

SOURCE_FILES: Final[tuple[str, ...]] = (
    "forge/cellular_symmetry/__init__.py",
    "forge/cellular_symmetry/__main__.py",
    "forge/cellular_symmetry/contract.py",
    "forge/cellular_symmetry/compiler.py",
    "forge/cellular_organism/compiler.py",
    "shared/schema/cellular_symmetry_bank.schema.json",
)


def source_sha256() -> str:
    digest = hashlib.sha256(b"nullvector-cellular-symmetry-source-v1\0")
    for dependency in (CELLULAR_CONTRACT_SHA256, breeding_source_sha256()):
        digest.update(dependency.encode("ascii")); digest.update(b"\0")
    for relative in SOURCE_FILES:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        digest.update(relative.encode("utf-8")); digest.update(b"\0")
        digest.update(path.read_bytes()); digest.update(b"\0")
    return digest.hexdigest()
