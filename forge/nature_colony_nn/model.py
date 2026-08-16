from __future__ import annotations

import torch
from torch import nn

from .contract import FEATURES,ROLES,ModelConfig


class ColonyCoordinator(nn.Module):
    def __init__(self,config:ModelConfig=ModelConfig())->None:
        super().__init__();self.config=config;self.input=nn.Sequential(nn.Linear(FEATURES,config.width),nn.LayerNorm(config.width),nn.SiLU());layer=nn.TransformerEncoderLayer(config.width,config.heads,config.width*4,config.dropout,batch_first=True,norm_first=True,activation="gelu");self.body=nn.TransformerEncoder(layer,config.layers,enable_nested_tensor=False);self.role=nn.Sequential(nn.LayerNorm(config.width),nn.Linear(config.width,len(ROLES)));self.action=nn.Sequential(nn.LayerNorm(config.width),nn.Linear(config.width,128),nn.SiLU(),nn.Linear(128,3),nn.Sigmoid())
    def forward(self,features:torch.Tensor,mask:torch.Tensor)->tuple[torch.Tensor,torch.Tensor]:
        hidden=self.body(self.input(features),src_key_padding_mask=~mask);return self.role(hidden),self.action(hidden)
