"""Loop-aware delta-focused successor for native-cell motion."""

from .contract import LoopTrainingConfig, source_sha256
from .evaluation import evaluate_checkpoint, evaluation_source_sha256, validate_evaluation
from .production import DEFAULT_OUTPUT, prepare_production, production_source_sha256, train_segment
from .sampler import LoopAwareRolloutBatchSampler
from .smoke import run_cpu_smoke, validate_cpu_smoke

__all__ = [
    "DEFAULT_OUTPUT",
    "LoopAwareRolloutBatchSampler",
    "LoopTrainingConfig",
    "evaluate_checkpoint",
    "evaluation_source_sha256",
    "prepare_production",
    "production_source_sha256",
    "run_cpu_smoke",
    "source_sha256",
    "train_segment",
    "validate_evaluation",
    "validate_cpu_smoke",
]
