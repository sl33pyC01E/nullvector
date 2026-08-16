from __future__ import annotations
import torch
from torch import nn
from .contract import ModelConfig

class LocalResidualBlock(nn.Module):
    def __init__(self,width):
        super().__init__();self.net=nn.Sequential(nn.Conv2d(width,width,3,padding=1),nn.GELU(),nn.Conv2d(width,width,3,padding=1))
    def forward(self,value):return value+self.net(value)*.15

class PixelCellRefiner(nn.Module):
    def __init__(self,config:ModelConfig=ModelConfig()):
        super().__init__();self.config=config;self.stem=nn.Conv2d(3,config.width,5,padding=2);self.blocks=nn.Sequential(*(LocalResidualBlock(config.width) for _ in range(config.blocks)));self.out=nn.Conv2d(config.width,3,3,padding=1);nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)
    def forward(self,base):
        delta=torch.tanh(self.out(self.blocks(torch.nn.functional.gelu(self.stem(base)))))*self.config.maximum_delta
        return torch.clamp(base+delta,0,1)
