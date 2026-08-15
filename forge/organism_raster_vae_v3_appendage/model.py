from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor,nn
import torch.nn.functional as F

from ..organism_raster_vae_v3.contract import RasterVAEV3Config
from ..organism_raster_vae_v3.model import StructuredRasterVAE,VAEOutput,loss as base_loss
from .dataset import APPENDAGE_CLASSES,INPUT_CHANNELS


@dataclass(slots=True)
class AppendageOutput(VAEOutput):
    appendage_logits: Tensor
    appendage_alpha_logits: Tensor


class AppendageRasterVAE(StructuredRasterVAE):
    def __init__(self,config: RasterVAEV3Config=RasterVAEV3Config()) -> None:
        super().__init__(config); w=config.base_width; self.stem=nn.Conv2d(INPUT_CHANNELS,w,3,padding=1); self.appendage=nn.Conv2d(w,len(APPENDAGE_CLASSES),1); self.render=nn.Sequential(nn.Conv2d(w//2,w//2,3,padding=1),nn.SiLU(),nn.Conv2d(w//2,5,1))

    def forward(self,living: Tensor,family: Tensor,traits: Tensor,phase: Tensor,*,generator: torch.Generator|None=None,stochastic: bool=True) -> AppendageOutput:
        condition=self.condition_vector(family,traits,phase); value=self.stem(living.float())
        for block in self.e48:value=block(value,condition)
        value=F.silu(self.d24(value))
        for block in self.e24:value=block(value,condition)
        fmu,flv=self.fmu(value),self.flv(value).clamp(-10,5); fine=self.sample(fmu,flv,generator,stochastic); value=F.silu(self.d12(value))
        for block in self.e12:value=block(value,condition)
        amu,alv=self.amu(value),self.alv(value).clamp(-10,5); anatomy=self.sample(amu,alv,generator,stochastic); value=F.silu(self.d6(value))
        for block in self.e6:value=block(value,condition)
        gmu,glv=self.gmu(value),self.glv(value).clamp(-10,5); global_latent=self.sample(gmu,glv,generator,stochastic); value=self.gin(global_latent)
        for block in self.x6:value=block(value,condition)
        value=self.afuse(torch.cat((self.u12(value),anatomy),1))
        for block in self.x12:value=block(value,condition)
        value=self.ffuse(torch.cat((self.u24(value),fine),1))
        for block in self.x24:value=block(value,condition)
        value=self.u48(value)
        for block in self.x48:value=block(value,condition)
        occupancy=self.occupancy(value); tissue=self.tissue(value); appendage=self.appendage(value); rendered=self.render(self.u96(value)); cell_alpha=F.interpolate(occupancy,size=(96,96),mode="nearest"); appendage_alpha=rendered[:,4:]; alpha=torch.sigmoid(rendered[:,3:4]+cell_alpha*1.25+appendage_alpha*.78); rgba=torch.cat((torch.sigmoid(rendered[:,:3]),alpha),1)
        return AppendageOutput(rgba,occupancy,tissue,fmu,flv,fine,amu,alv,anatomy,gmu,glv,global_latent,appendage,appendage_alpha)


def loss(output: AppendageOutput,batch: dict[str,Tensor],config: RasterVAEV3Config,beta_scale: float) -> tuple[Tensor,dict[str,float]]:
    base,metrics=base_loss(output,batch,config,beta_scale); occupancy=batch["occupancy"].float(); target=batch["appendage"].long(); weight=.05+occupancy*1.2+(target>0).float()*6.0; ce=F.cross_entropy(output.appendage_logits,target,reduction="none"); appendage_ce=(ce*weight).sum()/weight.sum(); limb=batch["appendage_alpha"].float(); limb_weight=.05+limb*11.0; limb_bce=F.binary_cross_entropy_with_logits(output.appendage_alpha_logits.float(),limb,reduction="none"); limb_bce=(limb_bce*limb_weight).sum()/limb_weight.sum(); probability=output.appendage_alpha_logits.float().sigmoid(); intersection=(probability*limb).sum((1,2,3)); limb_dice=1-((2*intersection+1)/(probability.sum((1,2,3))+limb.sum((1,2,3))+1)).mean(); final_limb_miss=((1-output.rgba[:,3:].float())*limb).sum()/limb.sum().clamp_min(1); total=base+1.0*appendage_ce+1.5*limb_bce+.9*limb_dice+1.4*final_limb_miss; metrics.update({"loss":float(total.detach()),"appendage_ce":float(appendage_ce.detach()),"appendage_alpha_bce":float(limb_bce.detach()),"appendage_dice":float(limb_dice.detach()),"final_limb_miss":float(final_limb_miss.detach())}); return total,metrics
