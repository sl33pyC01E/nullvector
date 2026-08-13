"""Foreground-factored neural map decorator v2 research contract."""

from .contract import (
    CALIBRATION_GATES,
    PRODUCTION_GATES,
    FactoredLossConfig,
    FactoredModelConfig,
    ForegroundPatchConfig,
    V2_CONTRACT_SHA256,
    V2TrainingConfig,
)
from .model import FactoredDecoratorOutput, FactoredDecoratorV2

__all__ = [
    "CALIBRATION_GATES",
    "PRODUCTION_GATES",
    "FactoredDecoratorOutput",
    "FactoredDecoratorV2",
    "FactoredLossConfig",
    "FactoredModelConfig",
    "ForegroundPatchConfig",
    "V2TrainingConfig",
    "V2_CONTRACT_SHA256",
]
