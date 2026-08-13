"""Deterministic lineage-aware fusion and mutation of neural sprite fields."""

from .genetics import FUSION_MODES, MUTATION_MODES, fuse_specimen
from .latent import LATENT_MODES, latent_fuse
from .model import FusionGenome, FusionSpecimen
from .rig import build_fusion_binding

__all__ = [
    "FUSION_MODES",
    "MUTATION_MODES",
    "FusionGenome",
    "FusionSpecimen",
    "LATENT_MODES",
    "build_fusion_binding",
    "fuse_specimen",
    "latent_fuse",
]
