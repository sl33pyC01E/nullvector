from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor,nn
import torch.nn.functional as F

from .contract import INPUT_CHANNELS,RasterVAEV3Config,TISSUE_CLASSES


class Block(nn.Module):
    def __init__(self,channels: int,condition_dim: int) -> None:
        super().__init__(); groups=min(32,channels)
        while channels%groups: groups-=1
        self.n1=nn.GroupNorm(groups,channels); self.c1=nn.Conv2d(channels,channels,3,padding=1)
        self.n2=nn.GroupNorm(groups,channels); self.c2=nn.Conv2d(channels,channels,3,padding=1)
        self.film=nn.Linear(condition_dim,channels*2); self.gate=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Conv2d(channels,max(24,channels//8),1),nn.SiLU(),nn.Conv2d(max(24,channels//8),channels,1),nn.Sigmoid())
    def forward(self,value: Tensor,condition: Tensor) -> Tensor:
        scale,shift=self.film(condition).chunk(2,1); hidden=self.n1(value)*(1+scale[:,:,None,None])+shift[:,:,None,None]
        hidden=self.c2(F.silu(self.n2(self.c1(F.silu(hidden))))); return value+hidden*self.gate(hidden)


class Up(nn.Module):
    def __init__(self,source: int,target: int) -> None:
        super().__init__(); self.net=nn.Sequential(nn.Conv2d(source,target*4,3,padding=1),nn.PixelShuffle(2),nn.SiLU(),nn.Conv2d(target,target,3,padding=1))
    def forward(self,value: Tensor) -> Tensor: return self.net(value)


@dataclass(slots=True)
class VAEOutput:
    rgba: Tensor; occupancy_logits: Tensor; tissue_logits: Tensor
    fine_mean: Tensor; fine_logvar: Tensor; fine: Tensor
    anatomy_mean: Tensor; anatomy_logvar: Tensor; anatomy: Tensor
    global_mean: Tensor; global_logvar: Tensor; global_latent: Tensor


class StructuredRasterVAE(nn.Module):
    def __init__(self,config: RasterVAEV3Config=RasterVAEV3Config()) -> None:
        super().__init__(); self.config=config; w,m,a,g=config.base_width,config.mid_width,config.anatomy_width,config.global_width
        self.family=nn.Embedding(5,32); self.trait=nn.Sequential(nn.Linear(15,64),nn.SiLU(),nn.Linear(64,64)); self.phase=nn.Sequential(nn.Linear(2,32),nn.SiLU(),nn.Linear(32,32)); self.condition=nn.Sequential(nn.Linear(128,config.condition_dim),nn.SiLU(),nn.Linear(config.condition_dim,config.condition_dim))
        self.stem=nn.Conv2d(INPUT_CHANNELS,w,3,padding=1); self.e48=nn.ModuleList(Block(w,config.condition_dim) for _ in range(config.depth))
        self.d24=nn.Conv2d(w,m,4,2,1); self.e24=nn.ModuleList(Block(m,config.condition_dim) for _ in range(config.depth)); self.fmu=nn.Conv2d(m,config.fine_channels,1); self.flv=nn.Conv2d(m,config.fine_channels,1)
        self.d12=nn.Conv2d(m,a,4,2,1); self.e12=nn.ModuleList(Block(a,config.condition_dim) for _ in range(config.depth)); self.amu=nn.Conv2d(a,config.anatomy_channels,1); self.alv=nn.Conv2d(a,config.anatomy_channels,1)
        self.d6=nn.Conv2d(a,g,4,2,1); self.e6=nn.ModuleList(Block(g,config.condition_dim) for _ in range(config.global_depth)); self.gmu=nn.Conv2d(g,config.global_channels,1); self.glv=nn.Conv2d(g,config.global_channels,1)
        self.gin=nn.Conv2d(config.global_channels,g,1); self.x6=nn.ModuleList(Block(g,config.condition_dim) for _ in range(config.global_depth)); self.u12=Up(g,a); self.afuse=nn.Conv2d(a+config.anatomy_channels,a,3,padding=1); self.x12=nn.ModuleList(Block(a,config.condition_dim) for _ in range(config.depth))
        self.u24=Up(a,m); self.ffuse=nn.Conv2d(m+config.fine_channels,m,3,padding=1); self.x24=nn.ModuleList(Block(m,config.condition_dim) for _ in range(config.depth)); self.u48=Up(m,w); self.x48=nn.ModuleList(Block(w,config.condition_dim) for _ in range(config.depth))
        self.occupancy=nn.Conv2d(w,1,1); self.tissue=nn.Conv2d(w,TISSUE_CLASSES,1); self.u96=Up(w,w//2); self.render=nn.Sequential(nn.Conv2d(w//2,w//2,3,padding=1),nn.SiLU(),nn.Conv2d(w//2,4,1))

    def condition_vector(self,family: Tensor,traits: Tensor,phase: Tensor) -> Tensor: return self.condition(torch.cat((self.family(family.long()),self.trait(traits.float()),self.phase(phase.float())),1))
    @staticmethod
    def sample(mean: Tensor,logvar: Tensor,generator: torch.Generator|None,stochastic: bool) -> Tensor:
        return mean if not stochastic else mean+torch.exp(.5*logvar)*torch.randn(mean.shape,device=mean.device,dtype=mean.dtype,generator=generator)
    def forward(self,living: Tensor,family: Tensor,traits: Tensor,phase: Tensor,*,generator: torch.Generator|None=None,stochastic: bool=True) -> VAEOutput:
        condition=self.condition_vector(family,traits,phase); value=self.stem(living.float())
        for block in self.e48:value=block(value,condition)
        value=F.silu(self.d24(value));
        for block in self.e24:value=block(value,condition)
        fmu,flv=self.fmu(value),self.flv(value).clamp(-10,5); fine=self.sample(fmu,flv,generator,stochastic)
        value=F.silu(self.d12(value));
        for block in self.e12:value=block(value,condition)
        amu,alv=self.amu(value),self.alv(value).clamp(-10,5); anatomy=self.sample(amu,alv,generator,stochastic)
        value=F.silu(self.d6(value));
        for block in self.e6:value=block(value,condition)
        gmu,glv=self.gmu(value),self.glv(value).clamp(-10,5); global_latent=self.sample(gmu,glv,generator,stochastic)
        value=self.gin(global_latent)
        for block in self.x6:value=block(value,condition)
        value=self.afuse(torch.cat((self.u12(value),anatomy),1));
        for block in self.x12:value=block(value,condition)
        value=self.ffuse(torch.cat((self.u24(value),fine),1));
        for block in self.x24:value=block(value,condition)
        value=self.u48(value)
        for block in self.x48:value=block(value,condition)
        occupancy=self.occupancy(value); tissue=self.tissue(value); rendered=self.render(self.u96(value))
        # The high-resolution alpha residual refines a learned cell-occupancy
        # scaffold.  This keeps one-cell limbs crisp without replacing the
        # neural rasterizer with deterministic post-processing.
        cell_alpha=F.interpolate(occupancy,size=(96,96),mode="nearest")
        rgba=torch.cat((torch.sigmoid(rendered[:,:3]),torch.sigmoid(rendered[:,3:]+cell_alpha*1.35)),1)
        return VAEOutput(rgba,occupancy,tissue,fmu,flv,fine,amu,alv,anatomy,gmu,glv,global_latent)


def _kl(mean: Tensor,logvar: Tensor,free_bits: float) -> Tensor: return (-.5*(1+logvar.float()-mean.float().square()-logvar.float().exp()).mean((2,3))).clamp_min(free_bits).mean()


def loss(output: VAEOutput,batch: dict[str,Tensor],config: RasterVAEV3Config,beta_scale: float) -> tuple[Tensor,dict[str,float]]:
    target=batch["rgba"].float(); alpha=target[:,3:]; weight=.12+2.88*alpha
    probability=output.rgba[:,3:].float().clamp(1e-5,1-1e-5)
    alpha_bce=(-(alpha*probability.log()+(1-alpha)*torch.log1p(-probability))*weight).sum()/weight.sum(); inter=(probability*alpha).sum((1,2,3)); dice=1-((2*inter+1)/(probability.sum((1,2,3))+alpha.sum((1,2,3))+1)).mean()
    rgb=((output.rgba[:,:3]-target[:,:3]).abs()*weight).sum()/(weight.sum()*3)
    occupancy=F.binary_cross_entropy_with_logits(output.occupancy_logits,batch["occupancy"].float()[:,None],pos_weight=torch.tensor(4.0,device=target.device))
    tissue=F.cross_entropy(output.tissue_logits,batch["tissue"].long(),reduction="none"); tissue=(tissue*(.1+batch["occupancy"].float()*2.9)).mean()
    # Multiscale edge loss directly penalizes the blurry/muddy failure mode.
    edge=sum((a-b).abs().mean() for a,b in ((output.rgba[:,:,:,1:]-output.rgba[:,:,:,:-1],target[:,:,:,1:]-target[:,:,:,:-1]),(output.rgba[:,:,1:,:]-output.rgba[:,:,:-1,:],target[:,:,1:,:]-target[:,:,:-1,:])))
    kf=_kl(output.fine_mean,output.fine_logvar,config.free_bits); ka=_kl(output.anatomy_mean,output.anatomy_logvar,config.free_bits); kg=_kl(output.global_mean,output.global_logvar,config.free_bits)
    reconstruction=2.5*alpha_bce+.8*dice+3.5*rgb+.75*occupancy+.65*tissue+.8*edge
    total=reconstruction+beta_scale*(config.beta_fine*kf+config.beta_anatomy*ka+config.beta_global*kg)
    return total,{"loss":float(total.detach()),"reconstruction":float(reconstruction.detach()),"alpha_bce":float(alpha_bce.detach()),"dice":float(dice.detach()),"rgb_l1":float(rgb.detach()),"occupancy_bce":float(occupancy.detach()),"tissue_ce":float(tissue.detach()),"edge_l1":float(edge.detach()),"kl_fine":float(kf.detach()),"kl_anatomy":float(ka.detach()),"kl_global":float(kg.detach())}
