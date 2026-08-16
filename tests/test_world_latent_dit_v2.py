from __future__ import annotations
import torch
from forge.world_latent_dit.model import ActionDiT
from forge.world_latent_dit.contract import ModelConfig

def test_residual_backbone_can_learn_spatial_delta_without_iterative_drift():
    model=ActionDiT(ModelConfig(width=128,layers=2,heads=4,patch=4));current=torch.randn(2,48,32,32);target=current+torch.randn_like(current)*.1;residual=model(current,torch.zeros(2),torch.arange(2),torch.zeros(2,4),torch.zeros(2,64));loss=(current+residual-target).square().mean();loss.backward();assert residual.shape==current.shape and torch.isfinite(loss) and model.patch.weight.grad is not None
