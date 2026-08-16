from __future__ import annotations

import numpy as np
import torch

from forge.world_action_step_v3.contract import ModelConfig
from forge.world_action_step_v3.data import align_causal_step
from forge.world_action_step_v3.runtime import WorldActionStepRuntime
from forge.world_latent_dit.contract import ModelConfig as BackboneConfig
from forge.world_latent_dit.model import ActionDiT


def test_causal_alignment_uses_next_command_and_current_state():
    latent = np.arange(5 * 2, dtype=np.float32).reshape(5, 2, 1, 1)
    raw = {
        "frame": np.arange(5 * 3, dtype=np.uint8).reshape(5, 1, 1, 3),
        "control": np.arange(20, dtype=np.float32).reshape(5, 4),
        "action": np.arange(5, dtype=np.uint8),
        "state": np.arange(5 * 64, dtype=np.float32).reshape(5, 64),
        "tick": np.arange(10, 15, dtype=np.int64),
    }
    aligned = align_causal_step(latent, raw)
    assert np.array_equal(aligned["current"], latent[:-1])
    assert np.array_equal(aligned["target"], latent[1:])
    assert np.array_equal(aligned["action"], raw["action"][1:])
    assert np.array_equal(aligned["control"], raw["control"][1:])
    assert np.array_equal(aligned["state"], raw["state"][:-1])


def test_runtime_preserves_latent_shape():
    config = ModelConfig(width=64, layers=1, heads=4, patch=4)
    model = ActionDiT(BackboneConfig(width=64, layers=1, heads=4, patch=4)).eval()
    runtime = WorldActionStepRuntime(model, torch.device("cpu"), {}, torch.zeros(1, 48, 1, 1), torch.ones(1, 48, 1, 1))
    current = torch.randn(2, 48, 32, 32)
    output = runtime.predict_latent(current, action=np.array([1, 2]), control=np.zeros((2, 4), np.float32), state=np.zeros((2, 64), np.float32))
    assert output.shape == current.shape
    assert torch.isfinite(output).all()
