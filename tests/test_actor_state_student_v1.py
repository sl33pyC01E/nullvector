from __future__ import annotations
import torch
from forge.actor_state_student_v1 import ActorStateStudent
def test_actor_student_shape_and_zero_initial_residual():
    model=ActorStateStudent(width=64);current=torch.randn(3,128);result=model(current,torch.randn_like(current),torch.tensor([0,1,2]),torch.zeros(3,4),torch.zeros(3,64));assert result.state.shape==(3,128);assert torch.equal(result.state,current);assert result.gate.shape==(3,128)
