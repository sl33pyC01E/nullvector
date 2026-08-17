from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from .contract import ModelConfig

class ResidualBlock(nn.Module):
    def __init__(self,width,dilation=1):
        super().__init__();self.norm1=nn.GroupNorm(16,width);self.conv1=nn.Conv2d(width,width,3,padding=dilation,dilation=dilation);self.norm2=nn.GroupNorm(16,width);self.conv2=nn.Conv2d(width,width,3,padding=1);self.condition=nn.Linear(width,width*2)
    def forward(self,x,condition):
        scale,bias=self.condition(condition).chunk(2,1);h=self.conv1(F.silu(self.norm1(x)));h=self.norm2(h)*(1+scale[:,:,None,None])+bias[:,:,None,None];return x+self.conv2(F.silu(h))

class WholeViewportLatentModel(nn.Module):
    """Numeric scene state to the sole full-frame VAE latent.

    This model never receives a sprite, tile, canvas, or rasterized cell. The
    only image-shaped inputs are semantic numeric fields owned by the ensemble.
    """
    def __init__(self,config=ModelConfig()):
        super().__init__();self.config=config;w=config.width
        input_channels=config.spatial_channels+config.actor_field_channels+2+config.latent_channels
        self.scene=nn.Conv2d(input_channels,w,3,padding=1)
        self.organism=nn.Sequential(nn.LayerNorm(config.organism_features),nn.Linear(config.organism_features,w),nn.SiLU(),nn.Linear(w,w))
        self.organism_mix=nn.Conv2d(w,w,3,padding=1)
        self.action=nn.Embedding(config.action_count,w)
        self.global_condition=nn.Sequential(nn.Linear(config.global_features,w*2),nn.SiLU(),nn.Linear(w*2,w))
        self.blocks=nn.ModuleList(ResidualBlock(w,(1,2,3,1)[index%4]) for index in range(config.blocks))
        self.out=nn.Sequential(nn.GroupNorm(16,w),nn.SiLU(),nn.Conv2d(w,config.latent_channels,3,padding=1))
        self.gate=nn.Sequential(nn.GroupNorm(16,w),nn.SiLU(),nn.Conv2d(w,config.latent_channels,3,padding=1))
        nn.init.zeros_(self.out[-1].weight);nn.init.zeros_(self.out[-1].bias)
        nn.init.zeros_(self.gate[-1].weight);nn.init.constant_(self.gate[-1].bias,-2.0)

    @staticmethod
    def _splat(tokens,positions,mask):
        batch,count,channels=tokens.shape;canvas=tokens.new_zeros((batch,channels,32*32));weights=tokens.new_zeros((batch,1,32*32));position=positions.clamp(-.5,.5-.0001);pixel=((position+.5)*32).long();index=(pixel[:,:,1]*32+pixel[:,:,0]).clamp(0,1023);valid=mask.to(tokens.dtype)
        canvas.scatter_add_(2,index[:,None].expand(-1,channels,-1),tokens.transpose(1,2)*valid[:,None]);weights.scatter_add_(2,index[:,None,:],valid[:,None]);return (canvas/weights.clamp_min(1)).reshape(batch,channels,32,32)

    def forward(self,previous_latent,spatial,organisms,organism_mask,state,actor_state,actor_field,visibility,memory,control,action):
        batch=spatial.shape[0]
        if spatial.shape[1:]!=(self.config.spatial_channels,32,32) or organisms.shape[1:]!=(64,self.config.organism_features):raise ValueError("whole-viewport model input drifted")
        if previous_latent.shape[1:]!=(self.config.latent_channels,32,32):raise ValueError("whole-viewport previous latent drifted")
        scene=torch.cat((previous_latent,spatial,actor_field,visibility,memory),1);x=self.scene(scene)
        organism=self.organism(organisms);x=x+self.organism_mix(self._splat(organism,organisms[:,:,:2],organism_mask))
        global_state=torch.cat((state,actor_state,control),1);condition=self.global_condition(global_state)+self.action(action.reshape(batch))
        for block in self.blocks:x=block(x,condition)
        return previous_latent+torch.sigmoid(self.gate(x))*self.out(x)
