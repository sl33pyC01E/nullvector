import numpy as np
import torch

from forge.neural_world_state_v1.contract import CONDITION_NAMES, CONTINUOUS_NAMES, WorldStateModelConfig
from forge.neural_world_state_v1.data import build_corpus
from forge.neural_world_state_v1.model import NeuralWorldStateVAE


def test_world_state_corpus_is_deterministic_and_multiscale() -> None:
    left = build_corpus(128, seed=73); right = build_corpus(128, seed=73)
    assert left.sha256 == right.sha256
    assert left.terrain.shape == left.city.shape == (128, 32, 32)
    assert left.continuous.shape == (128, len(CONTINUOUS_NAMES), 32, 32)
    assert left.condition.shape == (128, len(CONDITION_NAMES))
    assert np.array_equal(left.continuous, right.continuous)
    assert len(np.unique(left.city)) >= 7 and len(np.unique(left.terrain)) >= 5


def test_world_state_model_shapes_and_gradients() -> None:
    model = NeuralWorldStateVAE(WorldStateModelConfig(width=16, latent_channels=4, global_features=16)); batch = 2
    result = model(torch.zeros((batch, 32, 32), dtype=torch.long), torch.zeros((batch, 32, 32), dtype=torch.long), torch.zeros((batch, len(CONTINUOUS_NAMES), 32, 32)), torch.zeros((batch, len(CONDITION_NAMES))))
    assert result.terrain.shape == result.city.shape == (batch, 8, 32, 32)
    assert result.continuous.shape == (batch, len(CONTINUOUS_NAMES), 32, 32)
    assert result.spatial.shape == (batch, 4, 8, 8) and result.global_state.shape == (batch, 16)
    (result.terrain.mean() + result.city.mean() + result.continuous.mean() + result.condition.mean()).backward()
    assert all(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)
