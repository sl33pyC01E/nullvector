from __future__ import annotations

import hashlib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRODUCTION_MANIFEST = PROJECT_ROOT / "checkpoints" / "sprite_latent_production_v1_run3" / "production_manifest.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "neural_fusion_production_v1_run2"
FORMAT = "nullvector-production-neural-latent-fusion-v1"
FUSION_MODES = ("linear", "spatial_weave", "voronoi_mosaic", "radial_graft", "channel_crossover", "spectral_splice")
MUTATION_MODES = ("none", "latent_gaussian", "spatial_burst", "channel_phase", "donor_transplant", "phase_wave")
SOURCE_FILES = (
    "forge/neural_fusion_production/__init__.py",
    "forge/neural_fusion_production/__main__.py",
    "forge/neural_fusion_production/codec.py",
    "forge/neural_fusion_production/contract.py",
    "forge/neural_fusion_production/operators.py",
    "forge/neural_fusion_production/pilot.py",
    "forge/neural_fusion/genetics.py",
    "forge/neural_fusion/hashing.py",
    "forge/neural_fusion/model.py",
    "forge/neural_fusion/rig.py",
)


def production_fusion_source_hash() -> str:
    digest = hashlib.sha256(b"nullvector-production-neural-fusion-source-v1\0")
    for relative in SOURCE_FILES:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        digest.update(relative.encode("utf-8")); digest.update(b"\0"); digest.update(path.read_bytes()); digest.update(b"\0")
    return digest.hexdigest()
