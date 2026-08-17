from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import Tensor,nn

@dataclass(slots=True)
class ActorOutput:state:Tensor;gate:Tensor;delta:Tensor
class ActorStateStudent(nn.Module):
    def __init__(self,width:int=384,features:int=128,actions:int=22):
        super().__init__();self.width=width;self.features=features;self.actions=actions;self.action=nn.Embedding(actions,64);self.context=nn.Sequential(nn.Linear(4+64,128),nn.SiLU(),nn.Linear(128,64));self.body=nn.Sequential(nn.Linear(features*2+128,width),nn.LayerNorm(width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU());self.delta=nn.Linear(width,features);self.gate=nn.Linear(width,features);nn.init.zeros_(self.delta.weight);nn.init.zeros_(self.delta.bias);nn.init.constant_(self.gate.bias,-4)
    @property
    def parameter_count(self):return sum(p.numel() for p in self.parameters())
    def forward(self,current,previous,action,control,state):
        context=torch.cat((self.action(action),self.context(torch.cat((control.float(),state.float()),1))),1);hidden=self.body(torch.cat((current,current-previous,context),1));delta=self.delta(hidden);gate=torch.sigmoid(self.gate(hidden));return ActorOutput(current+gate*delta,gate,delta)
