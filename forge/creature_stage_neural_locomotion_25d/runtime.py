from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

from ..creature_stage_neural_grounded_controller.contract import APPENDAGE_KINDS
from ..nature_sim_v2.state import OrganismState
from .contract import MAX_APPENDAGES,MAX_MUSCLES
from .model import NeuralLocomotion25D
from .training import load_model


@dataclass(slots=True)
class RuntimeBank:
    appendage_indices:tuple[int,...]
    muscle_indices:tuple[int,...]
    previous_contact:np.ndarray
    hidden:torch.Tensor|None
    appendage_meta:torch.Tensor
    appendage_mask:torch.Tensor
    muscle_meta:torch.Tensor
    muscle_owner:torch.Tensor
    muscle_mask:torch.Tensor

@dataclass(slots=True)
class RuntimeState:
    phase:float
    global_static:torch.Tensor
    banks:tuple[RuntimeBank,...]


def _partition_appendages(organism)->tuple[tuple[tuple[int,...],tuple[int,...]],...]:
    muscles={index:[] for index in range(len(organism.genome.appendages))}
    for muscle_index,muscle in enumerate(organism.muscles):
        owner=int(muscle[2])
        if owner not in muscles:raise ValueError("2.5D muscle owner exceeded appendage census")
        muscles[owner].append(muscle_index)
    banks=[];appendages=[];indices=[]
    for appendage_index in range(len(organism.genome.appendages)):
        owned=muscles[appendage_index]
        if len(owned)>MAX_MUSCLES:raise ValueError("single appendage exceeded neural muscle bank")
        if appendages and (len(appendages)>=MAX_APPENDAGES or len(indices)+len(owned)>MAX_MUSCLES):banks.append((tuple(appendages),tuple(indices)));appendages=[];indices=[]
        appendages.append(appendage_index);indices.extend(owned)
    if appendages or not banks:banks.append((tuple(appendages),tuple(indices)))
    return tuple(banks)


def _bank_metadata(organism,appendage_indices,muscle_indices):
    appendage=np.zeros((MAX_APPENDAGES,16),np.float32);appendage_mask=np.zeros(MAX_APPENDAGES,np.bool_);lookup={global_index:local for local,global_index in enumerate(appendage_indices)}
    for local,global_index in enumerate(appendage_indices):
        gene=organism.genome.appendages[global_index]
        if gene.kind not in APPENDAGE_KINDS:raise ValueError(f"2.5D appendage kind drifted: {gene.kind}")
        appendage[local,APPENDAGE_KINDS.index(gene.kind)]=1;appendage[local,8:]=(float(gene.side),float(gene.segments)/5,math.sin(math.tau*gene.phase),math.cos(math.tau*gene.phase),float(gene.root_offset[0])/24,float(gene.root_offset[1])/24,float(gene.endpoint[0])/24,float(gene.endpoint[1])/24);appendage_mask[local]=True
    muscle=np.zeros((MAX_MUSCLES,8),np.float32);muscle_owner=np.zeros(MAX_MUSCLES,np.int64);muscle_mask=np.zeros(MAX_MUSCLES,np.bool_)
    for local,global_index in enumerate(muscle_indices):
        row=organism.muscles[global_index];owner=int(row[2]);joint=float(row[6]);gene=organism.genome.appendages[owner];muscle_owner[local]=lookup[owner];muscle[local]=(float(row[3]),float(row[4]),float(row[5]),joint/5,math.sin(math.tau*gene.phase),math.cos(math.tau*gene.phase),math.sin(math.tau*joint/5),math.cos(math.tau*joint/5));muscle_mask[local]=True
    return appendage,appendage_mask,muscle,muscle_owner,muscle_mask


class NeuralLocomotionRuntime:
    """Online recurrent contact/muscle policy for active ecology bodies."""

    def __init__(self,model:NeuralLocomotion25D,*,device:str|torch.device="cuda")->None:
        self.device=torch.device(device);self.model=model.to(self.device).eval();self.states:dict[int,RuntimeState]={}

    @classmethod
    def from_checkpoint(cls,path,*,device="cuda")->"NeuralLocomotionRuntime":
        model,_=load_model(path,device=device,ema=None);return cls(model,device=device)

    def _register(self,entity:OrganismState)->RuntimeState:
        organism=entity.body.organism
        static=np.concatenate((np.asarray(organism.genome.family_mix,np.float32),np.asarray(organism.genome.traits,np.float32)))[None]
        tensor=lambda value:torch.from_numpy(value).to(self.device)
        banks=[]
        for appendage_indices,muscle_indices in _partition_appendages(organism):
            appendage,appendage_mask,muscle,muscle_owner,muscle_mask=_bank_metadata(organism,appendage_indices,muscle_indices);banks.append(RuntimeBank(appendage_indices,muscle_indices,np.zeros(MAX_APPENDAGES,np.float32),None,tensor(appendage[None]),tensor(appendage_mask[None]),tensor(muscle[None]),tensor(muscle_owner[None]),tensor(muscle_mask[None])))
        state=RuntimeState(0.0,tensor(static),tuple(banks))
        self.states[entity.entity_id]=state;return state

    @torch.inference_mode()
    def step(self,entity:OrganismState,desired_velocity:np.ndarray,delta:float,time:float)->np.ndarray:
        state=self.states.get(entity.entity_id) or self._register(entity)
        speed=float(np.linalg.norm(desired_velocity));state.phase=(state.phase+delta*(.42+speed*.72))%1
        direction=np.asarray(desired_velocity,np.float32);contacts=np.zeros(len(entity.body.organism.genome.appendages),np.bool_);muscles=np.zeros(len(entity.body.organism.muscles),np.float32);velocities=[];weights=[]
        for bank in state.banks:
            dynamic=np.concatenate(((math.sin(math.tau*state.phase),math.cos(math.tau*state.phase)),direction,np.asarray(entity.velocity,np.float32),bank.previous_contact)).astype(np.float32)[None,None]
            result=self.model(state.global_static,bank.appendage_meta,bank.appendage_mask,bank.muscle_meta,bank.muscle_owner.long(),bank.muscle_mask,torch.from_numpy(dynamic).to(self.device),bank.hidden);bank.hidden=result.hidden.detach();probability=torch.sigmoid(result.contact_logits[0,0]).float().cpu().numpy();bank.previous_contact=(probability>=.5).astype(np.float32)
            if bank.appendage_indices:contacts[np.asarray(bank.appendage_indices)]=bank.previous_contact[:len(bank.appendage_indices)].astype(np.bool_)
            if bank.muscle_indices:muscles[np.asarray(bank.muscle_indices)]=result.muscle[0,0,:len(bank.muscle_indices)].float().cpu().numpy()
            velocities.append(result.velocity[0,0].float().cpu().numpy());weights.append(max(1,len(bank.appendage_indices)))
        entity.neural_contacts=contacts;entity.neural_muscles=muscles;predicted=np.average(np.stack(velocities),axis=0,weights=np.asarray(weights))
        # The network predicts the teacher's physical ground velocity. Preserve
        # analog player magnitude so a barely deflected stick is not full speed.
        magnitude=min(1.0,speed/max(float(np.linalg.norm(predicted)),1e-6))
        return predicted*magnitude

    def forget(self,entity_id:int)->None:self.states.pop(entity_id,None)
