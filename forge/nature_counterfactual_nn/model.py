from __future__ import annotations
import torch
from torch import nn
from .contract import ACTIONS,FEATURES,SEQUENCE,ModelConfig

class CounterfactualTransformer(nn.Module):
    def __init__(self,config:ModelConfig=ModelConfig()):
        super().__init__();self.config=config;self.input=nn.Linear(FEATURES,config.width);self.position=nn.Parameter(torch.randn(1,SEQUENCE,config.width)*.015);self.action=nn.Embedding(len(ACTIONS),config.width)
        layer=nn.TransformerEncoderLayer(config.width,config.heads,config.width*4,config.dropout,batch_first=True,norm_first=True,activation="gelu");self.encoder=nn.TransformerEncoder(layer,config.layers,norm=nn.LayerNorm(config.width));self.state=nn.Sequential(nn.Linear(config.width,config.width),nn.GELU(),nn.Linear(config.width,FEATURES));self.value=nn.Sequential(nn.Linear(config.width,config.width//2),nn.GELU(),nn.Linear(config.width//2,2))
    def forward(self,sequence,action):
        token=self.input(sequence)+self.position+self.action(action)[:,None,:];hidden=self.encoder(token)[:,-1];state=torch.sigmoid(self.state(hidden));value=torch.sigmoid(self.value(hidden));return state,value[:,0],value[:,1]
