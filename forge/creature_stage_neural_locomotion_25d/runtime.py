from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

from ..creature_stage_neural_grounded_controller.dataset import owner_metadata
from ..nature_sim_v2.state import OrganismState
from .contract import MAX_APPENDAGES,MAX_MUSCLES
from .data import _muscle_metadata
from .model import NeuralLocomotion25D
from .training import load_model


@dataclass(slots=True)
class RuntimeState:
    phase:float
    previous_contact:np.ndarray
    hidden:torch.Tensor|None
    global_static:torch.Tensor
    appendage_meta:torch.Tensor
    appendage_mask:torch.Tensor
    muscle_meta:torch.Tensor
    muscle_owner:torch.Tensor
    muscle_mask:torch.Tensor


class NeuralLocomotionRuntime:
    """Online recurrent contact/muscle policy for active ecology bodies."""

    def __init__(self,model:NeuralLocomotion25D,*,device:str|torch.device="cuda")->None:
        self.device=torch.device(device);self.model=model.to(self.device).eval();self.states:dict[int,RuntimeState]={}

    @classmethod
    def from_checkpoint(cls,path,*,device="cuda")->"NeuralLocomotionRuntime":
        model,_=load_model(path,device=device,ema=None);return cls(model,device=device)

    def _register(self,entity:OrganismState)->RuntimeState:
        organism=entity.body.organism;appendage,appendage_mask=owner_metadata(organism);muscle,muscle_owner,muscle_mask=_muscle_metadata(organism)
        static=np.concatenate((np.asarray(organism.genome.family_mix,np.float32),np.asarray(organism.genome.traits,np.float32)))[None]
        tensor=lambda value:torch.from_numpy(value).to(self.device)
        state=RuntimeState(0.0,np.zeros(MAX_APPENDAGES,np.float32),None,tensor(static),tensor(appendage[None]),tensor(appendage_mask[None]),tensor(muscle[None]),tensor(muscle_owner[None]),tensor(muscle_mask[None]))
        self.states[entity.entity_id]=state;return state

    @torch.inference_mode()
    def step(self,entity:OrganismState,desired_velocity:np.ndarray,delta:float,time:float)->np.ndarray:
        state=self.states.get(entity.entity_id) or self._register(entity)
        speed=float(np.linalg.norm(desired_velocity));state.phase=(state.phase+delta*(.42+speed*.72))%1
        direction=np.asarray(desired_velocity,np.float32)
        dynamic=np.concatenate(((math.sin(math.tau*state.phase),math.cos(math.tau*state.phase)),direction,np.asarray(entity.velocity,np.float32),state.previous_contact)).astype(np.float32)[None,None]
        result=self.model(state.global_static,state.appendage_meta,state.appendage_mask,state.muscle_meta,state.muscle_owner.long(),state.muscle_mask,torch.from_numpy(dynamic).to(self.device),state.hidden)
        state.hidden=result.hidden.detach();probability=torch.sigmoid(result.contact_logits[0,0]).float().cpu().numpy();state.previous_contact=(probability>=.5).astype(np.float32)
        entity.neural_contacts=state.previous_contact[:len(entity.body.organism.genome.appendages)].astype(np.bool_)
        entity.neural_muscles=result.muscle[0,0,:len(entity.body.organism.muscles)].float().cpu().numpy()
        predicted=result.velocity[0,0].float().cpu().numpy()
        # The network predicts the teacher's physical ground velocity. Preserve
        # analog player magnitude so a barely deflected stick is not full speed.
        magnitude=min(1.0,speed/max(float(np.linalg.norm(predicted)),1e-6))
        return predicted*magnitude

    def forget(self,entity_id:int)->None:self.states.pop(entity_id,None)

