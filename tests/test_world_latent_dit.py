from __future__ import annotations
import torch
from pathlib import Path
from forge.world_latent_dit import ActionDiT,ModelConfig,WorldActionDiTRuntime

def test_action_dit_is_spatial_action_conditioned_and_differentiable():
    model=ActionDiT(ModelConfig(width=128,layers=2,heads=4,patch=4));latent=torch.randn(3,48,32,32);time=torch.rand(3);action=torch.arange(3);control=torch.randn(3,4);state=torch.rand(3,64);result=model(latent,time,action,control,state);assert result.shape==latent.shape and torch.isfinite(result).all();result.square().mean().backward();assert model.action.weight.grad is not None and model.patch.weight.grad is not None

def test_promoted_action_dit_beats_persistence_and_runs_on_cpu():
    checkpoint=Path("game/generated/models/world_latent_dit/action_dit_v1.pt")
    if not checkpoint.is_file():return
    runtime=WorldActionDiTRuntime.from_checkpoint(checkpoint,device="cpu")
    assert runtime.report["latent_improvement"]>0 and runtime.report["rgb_improvement"]>0
    prediction=runtime.predict_latent(torch.zeros(1,48,32,32),action=0,control=[[0,0,0,0]],state=[[0]*64],steps=1)
    assert prediction.shape==(1,48,32,32) and torch.isfinite(prediction).all()
