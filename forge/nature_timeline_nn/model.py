from __future__ import annotations
import torch
from torch import nn
from .contract import EVENTS,FEATURES,SEQUENCE,ModelConfig

class TimelineTransformer(nn.Module):
    def __init__(self,config:ModelConfig=ModelConfig())->None:
        super().__init__();self.config=config;self.input=nn.Linear(FEATURES,config.width);self.position=nn.Parameter(torch.randn(SEQUENCE,config.width)*.015);layer=nn.TransformerEncoderLayer(config.width,config.heads,config.width*4,config.dropout,activation="gelu",batch_first=True,norm_first=True);self.body=nn.TransformerEncoder(layer,config.layers);self.norm=nn.LayerNorm(config.width);self.state=nn.Linear(config.width,FEATURES);self.event=nn.Linear(config.width,len(EVENTS));self.confidence=nn.Linear(config.width,1)
    def forward(self,x):
        hidden=self.norm(self.body(self.input(x)+self.position[None]));last=hidden[:,-1];return self.state(last),self.event(last),torch.sigmoid(self.confidence(last)).squeeze(-1)
