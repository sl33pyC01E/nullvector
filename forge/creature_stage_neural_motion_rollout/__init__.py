"""Prediction-fed successor training for native-cell creature motion."""

from .contract import DEFAULT_OUTPUT, RolloutTrainingConfig, source_sha256
from .evaluation import evaluate_checkpoint, validate_evaluation
from .training import prepare_production, run_cpu_smoke, train_segment, validate_cpu_smoke

__all__ = [
    "DEFAULT_OUTPUT",
    "RolloutTrainingConfig",
    "evaluate_checkpoint",
    "prepare_production",
    "run_cpu_smoke",
    "source_sha256",
    "train_segment",
    "validate_evaluation",
    "validate_cpu_smoke",
]
