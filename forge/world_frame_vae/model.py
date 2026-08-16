from __future__ import annotations
import torch
from torch import nn
from .contract import ModelConfig

class ResBlock(nn.Module):
    def __init__(self,channels:int):
        super().__init__();groups=max(1,min(32,channels//8));self.net=nn.Sequential(nn.GroupNorm(groups,channels),nn.SiLU(),nn.Conv2d(channels,channels,3,padding=1),nn.GroupNorm(groups,channels),nn.SiLU(),nn.Conv2d(channels,channels,3,padding=1))
    def forward(self,x):return x+self.net(x)

class Down(nn.Module):
    def __init__(self,source,target):super().__init__();self.block=ResBlock(source);self.down=nn.Conv2d(source,target,4,2,1)
    def forward(self,x):return self.down(self.block(x))

class Up(nn.Module):
    def __init__(self,source,target):super().__init__();self.conv=nn.Conv2d(source,target,3,padding=1);self.block=ResBlock(target)
    def forward(self,x):return self.block(self.conv(nn.functional.interpolate(x,scale_factor=2,mode="nearest")))

class WorldFrameVAE(nn.Module):
    def __init__(self,config:ModelConfig=ModelConfig()):
        super().__init__();self.config=config;b=config.base;channels=(b,b*3//2,b*2,b*3);self.stem=nn.Conv2d(3,channels[0],3,padding=1);self.encoder=nn.ModuleList(Down(channels[i],channels[i+1]) for i in range(3));self.mid=nn.Sequential(ResBlock(channels[-1]),ResBlock(channels[-1]),ResBlock(channels[-1]));self.statistics=nn.Conv2d(channels[-1],config.latent_channels*2,1);self.from_latent=nn.Conv2d(config.latent_channels,channels[-1],1);self.decoder=nn.ModuleList(Up(channels[i],channels[i-1]) for i in range(3,0,-1));self.out=nn.Sequential(ResBlock(channels[0]),nn.GroupNorm(16,channels[0]),nn.SiLU(),nn.Conv2d(channels[0],3,3,padding=1))
    def encode(self,x):
        hidden=self.stem(x)
        for block in self.encoder:hidden=block(hidden)
        mean,logvar=self.statistics(self.mid(hidden)).chunk(2,1);return mean,torch.clamp(logvar,-8,5)
    def decode(self,z):
        hidden=self.from_latent(z)
        for block in self.decoder:hidden=block(hidden)
        return torch.sigmoid(self.out(hidden))
    def forward(self,x,*,sample=True):
        mean,logvar=self.encode(x);z=mean+torch.randn_like(mean)*torch.exp(.5*logvar) if sample else mean;return self.decode(z),mean,logvar
