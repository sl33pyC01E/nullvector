"""Residual neural training over the immutable v4 public-proposal substrate."""

from .contract import ResidualLossConfig, ResidualTrainingConfig, V4_TRAINING_CONTRACT_SHA256
from .training import make_optimizer, train_batch

__all__ = [
    "ResidualLossConfig",
    "ResidualTrainingConfig",
    "V4_TRAINING_CONTRACT_SHA256",
    "make_optimizer",
    "train_batch",
]
