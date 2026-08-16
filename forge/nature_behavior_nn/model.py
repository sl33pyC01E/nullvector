from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import Tensor,nn

from ..nature_sim_v2.contract import INTENTS,RESOURCE_NAMES
from .contract import MAX_NEIGHBORS,NEIGHBOR_FEATURES,RESOURCE_FEATURES,SELF_FEATURES,ModelConfig


@dataclass(slots=True)
class BehaviorOutput:
    intent_logits:Tensor
    direction:Tensor
    urgency:Tensor


class NeuralNatureBehavior(nn.Module):
    """World-aware entity transformer used as the ecology action engine."""
    def __init__(self,config:ModelConfig=ModelConfig())->None:
        super().__init__();self.config=config;w=config.width
        self.self_encoder=nn.Sequential(nn.Linear(SELF_FEATURES,w),nn.LayerNorm(w),nn.SiLU(),nn.Linear(w,w))
        self.resource_encoder=nn.Sequential(nn.Linear(RESOURCE_FEATURES,w),nn.SiLU(),nn.Linear(w,w));self.resource_id=nn.Embedding(len(RESOURCE_NAMES),w)
        self.neighbor_encoder=nn.Sequential(nn.Linear(NEIGHBOR_FEATURES,w),nn.SiLU(),nn.Linear(w,w))
        self.type_embedding=nn.Embedding(3,w)
        layer=nn.TransformerEncoderLayer(w,config.heads,w*4,config.dropout,activation="gelu",batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(layer,config.layers,nn.LayerNorm(w))
        self.intent=nn.Sequential(nn.Linear(w,w),nn.SiLU(),nn.Linear(w,len(INTENTS)))
        self.direction=nn.Sequential(nn.Linear(w,w),nn.SiLU(),nn.Linear(w,2),nn.Tanh())
        self.urgency=nn.Sequential(nn.Linear(w,w//2),nn.SiLU(),nn.Linear(w//2,1))

    @property
    def parameter_count(self)->int:return sum(p.numel() for p in self.parameters())

    def forward(self,self_features:Tensor,resource:Tensor,neighbor:Tensor,neighbor_mask:Tensor)->BehaviorOutput:
        batch=self_features.shape[0];device=self_features.device
        self_token=self.self_encoder(self_features.float())[:,None]+self.type_embedding.weight[0][None,None]
        resource_id=torch.arange(resource.shape[1],device=device)[None].expand(batch,-1)
        resource_token=self.resource_encoder(resource.float())+self.resource_id(resource_id)+self.type_embedding.weight[1][None,None]
        neighbor_token=self.neighbor_encoder(neighbor.float())+self.type_embedding.weight[2][None,None]
        tokens=torch.cat((self_token,resource_token,neighbor_token),1)
        padding=torch.cat((torch.zeros((batch,1+resource.shape[1]),dtype=torch.bool,device=device),~neighbor_mask.bool()),1)
        encoded=self.encoder(tokens,src_key_padding_mask=padding)[:,0]
        return BehaviorOutput(self.intent(encoded),self.direction(encoded),self.urgency(encoded)[:,0])
