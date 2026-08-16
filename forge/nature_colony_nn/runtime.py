from __future__ import annotations

from pathlib import Path
import numpy as np
import torch

from ..nature_sim_v2.phenotype import phenotype_vector
from .contract import CHECKPOINT_FORMAT,MAX_MEMBERS,ROLES,ModelConfig,source_sha256
from .model import ColonyCoordinator


class NeuralColonyRuntime:
    def __init__(self,model,device):self.model,self.device=model,device;self.last_actions={};self.cache={};self.calls=0
    @classmethod
    def from_checkpoint(cls,path:Path,device:str="cuda"):
        target=torch.device(device if device=="cpu" or torch.cuda.is_available() else "cpu");payload=torch.load(path,map_location="cpu",weights_only=True)
        if payload["format"]!=CHECKPOINT_FORMAT or payload["source_sha256"]!=source_sha256():raise ValueError("colony checkpoint provenance drifted")
        model=ColonyCoordinator(ModelConfig(**payload["model_config"]));model.load_state_dict(payload["ema_state"],strict=True);return cls(model.to(target).eval(),target)
    @torch.inference_mode()
    def assign(self,members,state)->dict[int,str]:
        members=members[:MAX_MEMBERS];self.calls+=1;signature=tuple((member.entity_id,member.genome.developmental.generation,round(member.energy,1),round(member.body.systems()["integrity"],1)) for member in members);cached=self.cache.get(state.colony_id)
        if cached is not None and cached[0]==signature and self.calls-cached[1]<12:self.last_actions.update(cached[3]);return dict(cached[2])
        rows=[];count=max(1,len(members));
        for member in members:
            vector=phenotype_vector(member.genome);systems=np.asarray(tuple(member.body.systems().values()),np.float32);phase=((member.entity_id+member.genome.developmental.seed)%len(ROLES))/len(ROLES);row=np.concatenate((vector,np.eye(5,dtype=np.float32)[member.family],systems,np.asarray((member.energy,member.reserve,("embryo","juvenile","mature","senescent").index(member.stage)/3 if member.stage in ("embryo","juvenile","mature","senescent") else 0,count/MAX_MEMBERS,state.energy_store,state.cohesion,np.sin(phase*np.pi*2),np.cos(phase*np.pi*2)),np.float32)));rows.append(row)
        features=torch.zeros((1,MAX_MEMBERS,64),dtype=torch.float32,device=self.device);mask=torch.zeros((1,MAX_MEMBERS),dtype=torch.bool,device=self.device);features[0,:len(rows)]=torch.from_numpy(np.stack(rows)).to(self.device);mask[0,:len(rows)]=True;logits,actions=self.model(features,mask);labels=logits[0,:len(rows)].argmax(-1).cpu().tolist();values=actions[0,:len(rows)].float().cpu().numpy();action_map={member.entity_id:tuple(map(float,value)) for member,value in zip(members,values)};assignments={member.entity_id:ROLES[label] for member,label in zip(members,labels)};self.last_actions.update(action_map);self.cache[state.colony_id]=(signature,self.calls,dict(assignments),dict(action_map));return assignments

    def action(self,entity_id:int)->tuple[float,float,float]|None:return self.last_actions.get(entity_id)
