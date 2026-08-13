from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Final

from ..cellular_organism.contract import CELLULAR_CONTRACT_SHA256
from ..config import PROJECT_ROOT
from ..neural_fusion_production_evolution.contract import evolution_source_hash


FORMAT: Final[str] = "nullvector-evolved-cellular-organism-bank-v1"
SPECIES_FORMAT: Final[str] = "nullvector-evolved-cellular-organism-species-v1"
DEFAULT_SOURCE: Final[Path] = PROJECT_ROOT / "outputs/neural_fusion_production_evolution_v1_run2/production_evolution_manifest.json"
DEFAULT_OUTPUT: Final[Path] = PROJECT_ROOT / "outputs/evolved_cellular_organism_v1"
SCHEMA_PATH: Final[Path] = PROJECT_ROOT / "shared/schema/evolved_cellular_organism_bank.schema.json"
SOURCE_FILES: Final[tuple[str, ...]] = (
    "forge/evolved_cellular_organism/__init__.py",
    "forge/evolved_cellular_organism/__main__.py",
    "forge/evolved_cellular_organism/contract.py",
    "forge/evolved_cellular_organism/compiler.py",
    "forge/cellular_organism/compiler.py",
    "shared/schema/evolved_cellular_organism_bank.schema.json",
)


def source_sha256() -> str:
    digest = hashlib.sha256(b"nullvector-evolved-cellular-organism-source-v1\0")
    digest.update(CELLULAR_CONTRACT_SHA256.encode("ascii"))
    digest.update(b"\0")
    digest.update(evolution_source_hash().encode("ascii"))
    digest.update(b"\0")
    for relative in SOURCE_FILES:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
