from __future__ import annotations
import math
import torch
from torch import nn
from .contract import ACTIONS,CONTROL_FEATURES,LATENT_CHANNELS,LATENT_SIZE,STATE_FEATURES,ModelConfig

def _modulate(x,shift,scale):return x*(1+scale[:,None])+shift[:,None]

class DiTBlock(nn.Module):
    def __init__(self,width,heads):
        super().__init__();self.norm1=nn.LayerNorm(width,elementwise_affine=False);self.attention=nn.MultiheadAttention(width,heads,batch_first=True);self.norm2=nn.LayerNorm(width,elementwise_affine=False);self.mlp=nn.Sequential(nn.Linear(width,width*4),nn.GELU(),nn.Linear(width*4,width));self.modulation=nn.Sequential(nn.SiLU(),nn.Linear(width,width*6))
        nn.init.zeros_(self.modulation[-1].weight);nn.init.zeros_(self.modulation[-1].bias)
    def forward(self,x,condition):
        shift1,scale1,gate1,shift2,scale2,gate2=self.modulation(condition).chunk(6,1);value=_modulate(self.norm1(x),shift1,scale1);x=x+gate1[:,None]*self.attention(value,value,value,need_weights=False)[0];x=x+gate2[:,None]*self.mlp(_modulate(self.norm2(x),shift2,scale2));return x

class ActionDiT(nn.Module):
    def __init__(self,config:ModelConfig=ModelConfig()):
        super().__init__();self.config=config;tokens=(LATENT_SIZE//config.patch)**2;self.patch=nn.Conv2d(LATENT_CHANNELS,config.width,config.patch,config.patch);self.position=nn.Parameter(torch.randn(1,tokens,config.width)*.015);self.action=nn.Embedding(ACTIONS,config.width);self.control=nn.Linear(CONTROL_FEATURES,config.width);self.state=nn.Linear(STATE_FEATURES,config.width);self.time=nn.Sequential(nn.Linear(64,config.width),nn.SiLU(),nn.Linear(config.width,config.width));self.blocks=nn.ModuleList(DiTBlock(config.width,config.heads) for _ in range(config.layers));self.norm=nn.LayerNorm(config.width,elementwise_affine=False);self.final_mod=nn.Sequential(nn.SiLU(),nn.Linear(config.width,config.width*2));self.out=nn.Linear(config.width,LATENT_CHANNELS*config.patch*config.patch);nn.init.zeros_(self.final_mod[-1].weight);nn.init.zeros_(self.final_mod[-1].bias);nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)
    @staticmethod
    def time_embedding(t):
        frequency=torch.exp(torch.linspace(math.log(1),math.log(10000),32,device=t.device));angle=t[:,None]*frequency[None]*math.tau;return torch.cat((torch.sin(angle),torch.cos(angle)),1)
    def forward(self,latent,t,action,control,state):
        condition=self.time(self.time_embedding(t))+self.action(action)+self.control(control)+self.state(state);token=self.patch(latent).flatten(2).transpose(1,2)+self.position
        for block in self.blocks:token=block(token,condition)
        shift,scale=self.final_mod(condition).chunk(2,1);token=self.out(_modulate(self.norm(token),shift,scale));batch=latent.shape[0];patch=self.config.patch;side=LATENT_SIZE//patch;token=token.view(batch,side,side,LATENT_CHANNELS,patch,patch).permute(0,3,1,4,2,5).reshape(batch,LATENT_CHANNELS,LATENT_SIZE,LATENT_SIZE);return token
