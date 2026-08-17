from __future__ import annotations
import torch
from torch import Tensor,nn
from ..world_latent_dit.contract import ModelConfig
from ..world_latent_dit.model import ActionDiT


class RecurrentActionDiT(nn.Module):
    """Promoted Action-DiT with causal visual memory and actor-state context."""
    def __init__(self,config:ModelConfig=ModelConfig()):
        super().__init__();self.config=config;self.backbone=ActionDiT(config);self.history=nn.Conv2d(48,48,1);self.actor=nn.Sequential(nn.Linear(128,128),nn.SiLU(),nn.Linear(128,64));nn.init.zeros_(self.history.weight);nn.init.zeros_(self.history.bias);nn.init.zeros_(self.actor[-1].weight);nn.init.zeros_(self.actor[-1].bias)
    @property
    def parameter_count(self):return sum(parameter.numel() for parameter in self.parameters())
    def forward(self,current:Tensor,previous:Tensor,action:Tensor,control:Tensor,state:Tensor,actor_state:Tensor)->Tensor:
        remembered=current+self.history(current-previous);conditioned=state+self.actor(actor_state.float());time=torch.zeros(len(current),device=current.device,dtype=current.dtype);return self.backbone(remembered,time,action,control,conditioned)
