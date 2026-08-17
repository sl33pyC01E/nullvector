from pathlib import Path

import pytest
import torch

from forge.monolithic_world_model_v1.contract import DirectContextConfig
from forge.monolithic_world_model_v1.model import build_encoder


def test_direct_context_encoder_shapes_and_determinism() -> None:
    model_a = build_encoder(DirectContextConfig()).eval()
    model_b = build_encoder(DirectContextConfig()).eval()
    terrain = torch.zeros(2, 32, 32, dtype=torch.long)
    city = torch.zeros_like(terrain)
    continuous = torch.zeros(2, 7, 32, 32)
    condition = torch.zeros(2, 15)
    with torch.inference_mode():
        first = model_a(terrain, city, continuous, condition)
        second = model_b(terrain, city, continuous, condition)
    assert first.shape == (2, 64)
    assert torch.equal(first, second)


def test_direct_context_encoder_rejects_bad_shapes() -> None:
    model = build_encoder(DirectContextConfig())
    with pytest.raises(ValueError):
        model(torch.zeros(1, 16, 16, dtype=torch.long), torch.zeros(1, 16, 16, dtype=torch.long), torch.zeros(1, 7, 16, 16), torch.zeros(1, 15))
