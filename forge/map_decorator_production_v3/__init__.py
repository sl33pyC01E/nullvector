"""CPU-only sparse-localization foundation for neural map decoration v3."""

from .contract import LocatorLossConfig, LocatorModelConfig, LocatorTrainingConfig, V3_CHECKPOINT_FORMAT_VERSION, V3_CONTRACT_SHA256
from .model import SparseLocatorDecoratorV3, SparseLocatorOutput
from .pilot import RealCorpusPilotConfig, run_real_corpus_pilot, validate_real_corpus_pilot

__all__ = [
    "LocatorLossConfig",
    "LocatorModelConfig",
    "LocatorTrainingConfig",
    "SparseLocatorDecoratorV3",
    "SparseLocatorOutput",
    "RealCorpusPilotConfig",
    "run_real_corpus_pilot",
    "validate_real_corpus_pilot",
    "V3_CONTRACT_SHA256",
    "V3_CHECKPOINT_FORMAT_VERSION",
]
