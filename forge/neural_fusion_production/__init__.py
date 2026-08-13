"""Production learned-latent sprite fusion and mutation."""

from .contract import FUSION_MODES, MUTATION_MODES, production_fusion_source_hash
from .operators import production_latent_fuse

__all__ = ["FUSION_MODES", "MUTATION_MODES", "production_fusion_source_hash", "production_latent_fuse"]
