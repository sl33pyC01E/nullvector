"""Sparse-localization neural map decorator with a bounded CUDA calibration."""

from .contract import LocatorLossConfig, LocatorModelConfig, LocatorTrainingConfig, V3_CHECKPOINT_FORMAT_VERSION, V3_CONTRACT_SHA256
from .model import SparseLocatorDecoratorV3, SparseLocatorOutput
from .pilot import RealCorpusPilotConfig, run_real_corpus_pilot, validate_real_corpus_pilot
from .calibration import CalibrationConfig, supervise_calibration, validate_calibration

__all__ = [
    "LocatorLossConfig",
    "LocatorModelConfig",
    "LocatorTrainingConfig",
    "SparseLocatorDecoratorV3",
    "SparseLocatorOutput",
    "RealCorpusPilotConfig",
    "run_real_corpus_pilot",
    "validate_real_corpus_pilot",
    "CalibrationConfig",
    "supervise_calibration",
    "validate_calibration",
    "V3_CONTRACT_SHA256",
    "V3_CHECKPOINT_FORMAT_VERSION",
]
