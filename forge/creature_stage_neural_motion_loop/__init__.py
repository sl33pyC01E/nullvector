"""Loop-aware delta-focused successor for native-cell motion."""

from .contract import LoopTrainingConfig, source_sha256
from .sampler import LoopAwareRolloutBatchSampler
from .smoke import run_cpu_smoke, validate_cpu_smoke

__all__ = [
    "LoopAwareRolloutBatchSampler",
    "LoopTrainingConfig",
    "run_cpu_smoke",
    "source_sha256",
    "validate_cpu_smoke",
]
