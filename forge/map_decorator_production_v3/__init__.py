"""CPU-only sparse-localization foundation for neural map decoration v3."""

from .contract import LocatorLossConfig, LocatorModelConfig, LocatorTrainingConfig, V3_CHECKPOINT_FORMAT_VERSION, V3_CONTRACT_SHA256
from .model import SparseLocatorDecoratorV3, SparseLocatorOutput

__all__ = [
    "LocatorLossConfig",
    "LocatorModelConfig",
    "LocatorTrainingConfig",
    "SparseLocatorDecoratorV3",
    "SparseLocatorOutput",
    "V3_CONTRACT_SHA256",
    "V3_CHECKPOINT_FORMAT_VERSION",
]
