from __future__ import annotations

import hashlib
from pathlib import Path

from ..neural_fusion_production.contract import production_fusion_source_hash


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FOUNDERS = PROJECT_ROOT / "outputs" / "neural_fusion_production_v1_run2" / "production_fusion_manifest.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "neural_fusion_production_evolution_v1_run2"
FORMAT = "nullvector-production-neural-latent-evolution-v1"
SOURCE_FILES = (
    "forge/neural_fusion_production_evolution/__init__.py",
    "forge/neural_fusion_production_evolution/__main__.py",
    "forge/neural_fusion_production_evolution/contract.py",
    "forge/neural_fusion_production_evolution/evolution.py",
)


def evolution_source_hash() -> str:
    digest = hashlib.sha256(b"nullvector-production-latent-evolution-source-v1\0")
    digest.update(production_fusion_source_hash().encode("ascii"))
    digest.update(b"\0")
    for relative in SOURCE_FILES:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        digest.update(relative.encode("utf-8")); digest.update(b"\0")
        digest.update(path.read_bytes()); digest.update(b"\0")
    return digest.hexdigest()
