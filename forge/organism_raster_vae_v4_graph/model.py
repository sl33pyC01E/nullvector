from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor,nn
import torch.nn.functional as F

from ..organism_raster_vae_v3.contract import RasterVAEV3Config
from ..organism_raster_vae_v3.model import StructuredRasterVAE,VAEOutput,loss as base_loss
from .dataset import MAX_TOKENS,TOKEN_FEATURES


@dataclass(slots=True)
class GraphOutput(VAEOutput):
    token_attention: Tensor


class GraphTokenRasterVAE(StructuredRasterVAE):
    def __init__(self,config: RasterVAEV3Config=RasterVAEV3Config()) -> None:
        super().__init__(config);self.token_embed=nn.Sequential(nn.Linear(TOKEN_FEATURES,256),nn.SiLU(),nn.Linear(256,256));self.token12=nn.Linear(256,config.anatomy_width);self.position12=nn.Linear(4,config.anatomy_width);self.position24=nn.Linear(4,config.mid_width);self.attention12=nn.MultiheadAttention(config.anatomy_width,8,batch_first=True);self.attention24=nn.MultiheadAttention(config.mid_width,8,batch_first=True);self.gate12=nn.Parameter(torch.zeros(()));self.gate24=nn.Parameter(torch.zeros(()))

    @staticmethod
    def _position(size: int,device: torch.device,dtype: torch.dtype) -> Tensor:
        axis=torch.linspace(-1,1,size,device=device,dtype=dtype);yy,xx=torch.meshgrid(axis,axis,indexing="ij");return torch.stack((torch.sin(torch.pi*xx),torch.cos(torch.pi*xx),torch.sin(torch.pi*yy),torch.cos(torch.pi*yy)),-1).reshape(1,size*size,4)

    def _attend(self,value: Tensor,tokens: Tensor,mask: Tensor,projection: nn.Linear,attention: nn.MultiheadAttention,position: nn.Linear,gate: Tensor) -> tuple[Tensor,Tensor]:
        batch,channels,height,width=value.shape;query=value.flatten(2).transpose(1,2)+position(self._position(height,value.device,value.dtype));keys=projection(tokens);result,weights=attention(query,keys,keys,key_padding_mask=~mask.bool(),need_weights=True,average_attn_weights=True);query=query+torch.tanh(gate)*result;return query.transpose(1,2).reshape(batch,channels,height,width),weights

    def forward(self,living: Tensor,family: Tensor,traits: Tensor,phase: Tensor,tokens: Tensor,token_mask: Tensor,*,generator: torch.Generator|None=None,stochastic: bool=True) -> GraphOutput:
        if tokens.shape[1:]!=(MAX_TOKENS,TOKEN_FEATURES) or token_mask.shape!=tokens.shape[:2]:raise ValueError("graph token conditioning drifted")
        condition=self.condition_vector(family,traits,phase);token=self.token_embed(tokens.float());value=self.stem(living.float())
        for block in self.e48:value=block(value,condition)
        value=F.silu(self.d24(value))
        for block in self.e24:value=block(value,condition)
        fmu,flv=self.fmu(value),self.flv(value).clamp(-10,5);fine=self.sample(fmu,flv,generator,stochastic);value=F.silu(self.d12(value))
        for block in self.e12:value=block(value,condition)
        amu,alv=self.amu(value),self.alv(value).clamp(-10,5);anatomy=self.sample(amu,alv,generator,stochastic);value=F.silu(self.d6(value))
        for block in self.e6:value=block(value,condition)
        gmu,glv=self.gmu(value),self.glv(value).clamp(-10,5);global_latent=self.sample(gmu,glv,generator,stochastic);value=self.gin(global_latent)
        for block in self.x6:value=block(value,condition)
        value=self.afuse(torch.cat((self.u12(value),anatomy),1));value,_=self._attend(value,token,token_mask,self.token12,self.attention12,self.position12,self.gate12)
        for block in self.x12:value=block(value,condition)
        value=self.ffuse(torch.cat((self.u24(value),fine),1));value,weights=self._attend(value,token,token_mask,nn.Identity(),self.attention24,self.position24,self.gate24)
        for block in self.x24:value=block(value,condition)
        value=self.u48(value)
        for block in self.x48:value=block(value,condition)
        occupancy=self.occupancy(value);tissue=self.tissue(value);rendered=self.render(self.u96(value));cell_alpha=F.interpolate(occupancy,size=(96,96),mode="nearest");rgba=torch.cat((torch.sigmoid(rendered[:,:3]),torch.sigmoid(rendered[:,3:]+cell_alpha*1.35)),1);return GraphOutput(rgba,occupancy,tissue,fmu,flv,fine,amu,alv,anatomy,gmu,glv,global_latent,weights)


def loss(output: GraphOutput,batch: dict[str,Tensor],config: RasterVAEV3Config,beta_scale: float) -> tuple[Tensor,dict[str,float]]:
    base,metrics=base_loss(output,batch,config,beta_scale);owner=batch["token_owner"][:,::2,::2].reshape(len(output.rgba),-1);valid=owner>=0;attention=output.token_attention.clamp_min(1e-7);selected=-attention.gather(2,owner.clamp_min(0)[:,:,None]).squeeze(2).log();owner_nll=selected[valid].mean() if bool(valid.any()) else selected.sum()*0;total=base+.45*owner_nll;metrics.update({"loss":float(total.detach()),"token_owner_nll":float(owner_nll.detach())});return total,metrics
