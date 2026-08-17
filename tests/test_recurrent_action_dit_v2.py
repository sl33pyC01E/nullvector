from __future__ import annotations
import torch
from forge.recurrent_action_dit_v2.model import RecurrentActionDiT
from forge.world_latent_dit.model import ActionDiT

def test_zero_memory_adapters_preserve_parent_backbone() -> None:
    torch.manual_seed(7);parent=ActionDiT();student=RecurrentActionDiT();student.backbone.load_state_dict(parent.state_dict());current=torch.randn(2,48,32,32);previous=torch.randn_like(current);action=torch.tensor((1,4));control=torch.randn(2,4);state=torch.randn(2,64);actor=torch.randn(2,128);time=torch.zeros(2)
    with torch.inference_mode():expected=parent(current,time,action,control,state);actual=student(current,previous,action,control,state,actor)
    assert torch.equal(expected,actual)
