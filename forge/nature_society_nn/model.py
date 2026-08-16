from __future__ import annotations

import torch
from torch import nn

from .contract import ACTIVITIES,DIPLOMACY,FEATURES,LABOR_SECTORS,PROJECTS,ModelConfig


class ResidualBlock(nn.Module):
    def __init__(self,width:int,dropout:float)->None:
        super().__init__();self.norm=nn.LayerNorm(width);self.net=nn.Sequential(nn.Linear(width,width*3),nn.SiLU(),nn.Dropout(dropout),nn.Linear(width*3,width))
    def forward(self,value:torch.Tensor)->torch.Tensor:return value+self.net(self.norm(value))


class SocietyStrategist(nn.Module):
    def __init__(self,config:ModelConfig=ModelConfig())->None:
        super().__init__();self.config=config;self.input=nn.Sequential(nn.Linear(FEATURES,config.width),nn.LayerNorm(config.width),nn.SiLU());self.body=nn.Sequential(*(ResidualBlock(config.width,config.dropout) for _ in range(config.depth)));self.final=nn.LayerNorm(config.width);self.activity=nn.Linear(config.width,len(ACTIVITIES));self.labor=nn.Linear(config.width,len(LABOR_SECTORS));self.diplomacy=nn.Linear(config.width,len(DIPLOMACY));self.project=nn.Linear(config.width,len(PROJECTS))
    def forward(self,features:torch.Tensor)->tuple[torch.Tensor,...]:
        hidden=self.final(self.body(self.input(features)));return self.activity(hidden),self.labor(hidden),self.diplomacy(hidden),self.project(hidden)
