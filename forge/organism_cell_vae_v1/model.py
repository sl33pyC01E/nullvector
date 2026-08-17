from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor,nn
import torch.nn.functional as F

from .contract import CELL_FEATURES


@dataclass(slots=True)
class CellVAEOutput:
    rgba:Tensor;mean:Tensor;logvar:Tensor;latent:Tensor;cell_rgba:Tensor;offset:Tensor;sigma:Tensor


class ContinuousCellVAE(nn.Module):
    def __init__(self,width:int=96,latent_dim:int=48)->None:
        super().__init__();self.width=width;self.latent_dim=latent_dim
        self.cell_encoder=nn.Sequential(nn.Linear(CELL_FEATURES,width),nn.SiLU(),nn.Linear(width,width),nn.LayerNorm(width));self.global_encoder=nn.Sequential(nn.Linear(width*2,width*2),nn.SiLU(),nn.Linear(width*2,latent_dim*2))
        self.decoder=nn.Sequential(nn.Linear(CELL_FEATURES+latent_dim,width*2),nn.SiLU(),nn.Linear(width*2,width*2),nn.SiLU(),nn.Linear(width*2,7));self.polish=nn.Sequential(nn.Conv2d(4,32,3,padding=1),nn.SiLU(),nn.Conv2d(32,32,3,padding=1),nn.SiLU(),nn.Conv2d(32,4,1));nn.init.zeros_(self.polish[-1].weight);nn.init.zeros_(self.polish[-1].bias)
        yy,xx=torch.meshgrid(torch.arange(96,dtype=torch.float32),torch.arange(96,dtype=torch.float32),indexing="ij");self.register_buffer("grid_x",xx[None,None],persistent=False);self.register_buffer("grid_y",yy[None,None],persistent=False)

    @staticmethod
    def sample(mean:Tensor,logvar:Tensor,generator:torch.Generator|None,stochastic:bool)->Tensor:return mean if not stochastic else mean+torch.exp(.5*logvar)*torch.randn(mean.shape,device=mean.device,dtype=mean.dtype,generator=generator)

    def _render(self,xy:Tensor,cell:Tensor,mask:Tensor)->Tensor:
        batch,count,_=xy.shape;density=torch.zeros(batch,1,96,96,device=xy.device);color=torch.zeros(batch,3,96,96,device=xy.device)
        rgb=torch.sigmoid(cell[:,:,:3]);strength=1.5+6*torch.sigmoid(cell[:,:,3:4]);sigma=.32+.48*torch.sigmoid(cell[:,:,4:5]);offset=.65*torch.tanh(cell[:,:,5:7]);position=xy+offset
        for start in range(0,count,72):
            stop=min(start+72,count);active=mask[:,start:stop,None,None].float();px=position[:,start:stop,0,None,None];py=position[:,start:stop,1,None,None];spread=sigma[:,start:stop,:,None];weight=torch.exp(-((self.grid_x-px).square()+(self.grid_y-py).square())/(2*spread.square()))*strength[:,start:stop,:,None]*active
            density=density+weight.sum(1,keepdim=True);color=color+(weight[:,:,None]*rgb[:,start:stop,:,None,None]).sum(1)
        alpha=1-torch.exp(-density);base_rgb=color/density.clamp_min(1e-5);base_rgb=base_rgb*(1-torch.exp(-4*density));base=torch.cat((base_rgb,alpha),1);delta=self.polish(base);out_rgb=torch.clamp(base_rgb+.18*torch.tanh(delta[:,:3]),0,1);out_alpha=torch.sigmoid(torch.logit(alpha.clamp(1e-5,1-1e-5))+delta[:,3:]);return torch.cat((out_rgb,out_alpha),1),offset,sigma

    def forward(self,features:Tensor,mask:Tensor,*,generator:torch.Generator|None=None,stochastic:bool=True)->CellVAEOutput:
        if features.ndim!=3 or features.shape[-1]!=CELL_FEATURES or mask.shape!=features.shape[:2] or not bool(mask.any(1).all()):raise ValueError("continuous cell VAE input drifted")
        encoded=self.cell_encoder(features.float());active=mask[:,:,None].float();mean_pool=(encoded*active).sum(1)/active.sum(1).clamp_min(1);max_pool=encoded.masked_fill(~mask[:,:,None],-1e4).max(1).values;mean,logvar=self.global_encoder(torch.cat((mean_pool,max_pool),1)).chunk(2,1);logvar=logvar.clamp(-10,5);latent=self.sample(mean,logvar,generator,stochastic);cell=self.decoder(torch.cat((features.float(),latent[:,None].expand(-1,len(features[0]),-1)),2));rgba,offset,sigma=self._render((features[:,:,:2].float()+1)*47*.5*2+.5,cell,mask);return CellVAEOutput(rgba,mean,logvar,latent,cell,offset,sigma)


def loss(output:CellVAEOutput,target:Tensor,mask:Tensor,beta:float)->tuple[Tensor,dict[str,float]]:
    target=target.float();alpha=target[:,3:];pa=output.rgba[:,3:].clamp(1e-5,1-1e-5);weight=.12+2.88*alpha;bce=(-(alpha*pa.log()+(1-alpha)*torch.log1p(-pa))*weight).sum()/weight.sum();inter=(pa*alpha).sum((1,2,3));dice=1-((2*inter+1)/(pa.sum((1,2,3))+alpha.sum((1,2,3))+1)).mean();rgb=((output.rgba[:,:3]-target[:,:3]).abs()*weight).sum()/(weight.sum()*3);edge=(pa[:,:,1:]-pa[:,:,:-1]-(alpha[:,:,1:]-alpha[:,:,:-1])).abs().mean()+(pa[:,:,:,1:]-pa[:,:,:,:-1]-(alpha[:,:,:,1:]-alpha[:,:,:,:-1])).abs().mean();kl=(-.5*(1+output.logvar-output.mean.square()-output.logvar.exp())).mean();offset=(output.offset.square()*mask[:,:,None]).sum()/mask.sum().clamp_min(1);total=3*bce+1.5*dice+3*rgb+1.2*edge+beta*2e-4*kl+1e-3*offset;return total,{"loss":float(total.detach()),"alpha_bce":float(bce.detach()),"dice":float(dice.detach()),"rgb_l1":float(rgb.detach()),"edge_l1":float(edge.detach()),"kl":float(kl.detach()),"offset_l2":float(offset.detach())}
