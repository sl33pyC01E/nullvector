"""Second-generation masked topology prior foundation.

This package is additive.  It deliberately does not alter the frozen v1 model,
checkpoints, corpus, or generation bank.
"""

from .conditioning import CONDITION_CHANNELS, build_spatial_conditions
from .contract import PRIOR_V2_FORMAT, PriorV2Config, prior_v2_source_sha256
from .masking import MASK_MODES_V2, mask_tokens_v2
from .model import MultiScaleTopologyPrior, build_prior_v2, masked_token_loss_v2

__all__ = [
    "CONDITION_CHANNELS",
    "MASK_MODES_V2",
    "PRIOR_V2_FORMAT",
    "MultiScaleTopologyPrior",
    "PriorV2Config",
    "build_prior_v2",
    "build_spatial_conditions",
    "mask_tokens_v2",
    "masked_token_loss_v2",
    "prior_v2_source_sha256",
]
