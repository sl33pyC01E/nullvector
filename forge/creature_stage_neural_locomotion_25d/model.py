from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import Tensor,nn

from .contract import APPENDAGE_FEATURES,DYNAMIC_FEATURES,MAX_APPENDAGES,MAX_MUSCLES,MUSCLE_FEATURES,ModelConfig


@dataclass(slots=True)
class LocomotionOutput:
    contact_logits:Tensor
    muscle:Tensor
    velocity:Tensor
    hidden:Tensor


class NeuralLocomotion25D(nn.Module):
    def __init__(self,config:ModelConfig=ModelConfig())->None:
        super().__init__();self.config=config;w=config.width
        self.static=nn.Sequential(nn.Linear(20,w),nn.SiLU(),nn.Linear(w,w))
        self.appendage=nn.Sequential(nn.Linear(APPENDAGE_FEATURES,w),nn.SiLU(),nn.Linear(w,w))
        self.dynamic=nn.Sequential(nn.Linear(DYNAMIC_FEATURES,w),nn.SiLU(),nn.Linear(w,w))
        self.gru=nn.GRU(w*2,w,num_layers=config.recurrent_layers,batch_first=True,dropout=config.dropout)
        self.contact_sequence=nn.Linear(w,w);self.contact_appendage=nn.Linear(w,w)
        self.contact=nn.Sequential(nn.SiLU(),nn.Linear(w,1))
        self.muscle_sequence=nn.Linear(w,w);self.muscle_appendage=nn.Linear(w,w);self.muscle_meta=nn.Linear(MUSCLE_FEATURES,w)
        self.muscle=nn.Sequential(nn.SiLU(),nn.Linear(w,w),nn.SiLU(),nn.Linear(w,1))
        self.velocity=nn.Sequential(nn.LayerNorm(w),nn.Linear(w,w),nn.SiLU(),nn.Linear(w,2),nn.Tanh())

    @property
    def parameter_count(self)->int:return sum(p.numel() for p in self.parameters())

    def forward(self,global_static:Tensor,appendage_meta:Tensor,appendage_mask:Tensor,muscle_meta:Tensor,muscle_owner:Tensor,muscle_mask:Tensor,dynamic:Tensor,hidden:Tensor|None=None)->LocomotionOutput:
        b,t,_=dynamic.shape
        static=self.static(global_static.float());appendage=self.appendage(appendage_meta.float())*appendage_mask[:,:,None]
        recurrent_input=torch.cat((self.dynamic(dynamic.float()),static[:,None].expand(-1,t,-1)),dim=-1)
        sequence,hidden_out=self.gru(recurrent_input,hidden)
        contact_input=self.contact_sequence(sequence)[:,:,None]+self.contact_appendage(appendage)[:,None]
        contact=self.contact(contact_input)[...,0].masked_fill(~appendage_mask[:,None],-30)
        owners=muscle_owner.clamp(0,MAX_APPENDAGES-1)
        gathered=torch.gather(appendage,1,owners[:,:,None].expand(-1,-1,appendage.shape[-1]))
        muscle_input=self.muscle_sequence(sequence)[:,:,None]+self.muscle_appendage(gathered)[:,None]+self.muscle_meta(muscle_meta.float())[:,None]
        muscle=torch.sigmoid(self.muscle(muscle_input)[...,0])*muscle_mask[:,None]
        velocity=self.velocity(sequence)*3.2
        return LocomotionOutput(contact,muscle,velocity,hidden_out)
