from __future__ import annotations

import torch

from forge.organism_raster_vae_v7_refiner.contract import Plan
from forge.organism_raster_vae_v7_refiner.model import NeuralCellRefiner,loss


def test_refiner_is_identity_at_initialization_and_differentiable()->None:
    model=NeuralCellRefiner(width=32);living=torch.rand(2,42,48,48);parent=torch.rand(2,4,96,96);output=model(living,parent)
    assert output.rgba.shape==parent.shape and torch.allclose(output.rgba,parent,atol=1e-5)
    target=torch.rand_like(parent);appendage=(target[:,3:]>.8).float();value,metrics=loss(output,target,appendage);value.backward()
    assert torch.isfinite(value) and model.out.weight.grad is not None and set(metrics)>={"alpha_bce","dice","rgb_l1"}


def test_plan_rejects_non_atomic_schedule()->None:
    try:Plan(total_steps=750,segment_steps=200)
    except ValueError:return
    raise AssertionError("invalid V7 schedule accepted")
