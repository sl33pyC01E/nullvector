from __future__ import annotations

import torch

from forge.aligned_world_monolith_v1 import AlignedWorldMonolith, ModelConfig
from forge.whole_viewport_latent_v1.contract import ModelConfig as RendererConfig


def _inputs(batch=2):
    return (
        torch.zeros(batch, 48, 32, 32), torch.zeros(batch, 68, 32, 32),
        torch.zeros(batch, 64, 164), torch.zeros(batch, 64, dtype=torch.bool),
        torch.zeros(batch, 64), torch.zeros(batch, 128), torch.zeros(batch, 8, 32, 32),
        torch.zeros(batch, 1, 32, 32), torch.zeros(batch, 1, 32, 32),
        torch.zeros(batch, 4), torch.zeros(batch, dtype=torch.long), torch.zeros(batch, 3),
        torch.zeros(batch, 5, 4),
    )


def test_aligned_monolith_starts_as_a_stable_physical_transition():
    model = AlignedWorldMonolith(ModelConfig(width=64, blocks=2, organism_width=96), RendererConfig(width=64, blocks=2))
    values = _inputs()
    with torch.no_grad():
        step = model.transition(*values)
    assert torch.count_nonzero(step.spatial) == 0
    assert torch.count_nonzero(step.organisms) == 0
    assert torch.count_nonzero(step.state) == 0
    assert torch.count_nonzero(step.actor_state) == 0
    assert torch.count_nonzero(step.actor_field) == 0
    assert tuple(step.organism_probability.shape) == (2, 64)
    assert float(step.organism_probability.max()) < .01


def test_aligned_monolith_complete_output_contract():
    model = AlignedWorldMonolith(ModelConfig(width=64, blocks=1, organism_width=96), RendererConfig(width=64, blocks=1))
    step = model(*_inputs(batch=1))
    assert tuple(step.latent.shape) == (1, 48, 32, 32)
    assert tuple(step.spatial.shape) == (1, 68, 32, 32)
    assert tuple(step.organisms.shape) == (1, 64, 164)
    assert tuple(step.state.shape) == (1, 64)
    assert tuple(step.actor_state.shape) == (1, 128)
    assert tuple(step.timeline_event.shape) == (1, 10)
    assert tuple(step.counterfactual.shape) == (1, 5, 4)
