from __future__ import annotations
from pathlib import Path
import numpy as np,torch
from ..world_latent_dit.model import ActionDiT
from ..world_latent_dit.contract import ModelConfig as BackboneConfig
from .contract import CHECKPOINT_FORMAT,source_sha256

class ResidualWorldActionDiTRuntime:
    def __init__(self,model,device,report,mean,std):self.model=model;self.device=device;self.report=report;self.mean=mean;self.std=std
    @classmethod
    def from_checkpoint(cls,path:Path,*,device="cuda"):
        target=torch.device(device if device!="cuda" or torch.cuda.is_available() else "cpu");payload=torch.load(Path(path),map_location="cpu",weights_only=False)
        if payload.get("format")!=CHECKPOINT_FORMAT or payload.get("source_sha256")!=source_sha256():raise ValueError("residual world action DiT checkpoint provenance drifted")
        model=ActionDiT(BackboneConfig(**payload["model_config"]));model.load_state_dict(payload["ema_state"]);model.to(target).eval();mean=torch.as_tensor(payload["latent_mean"],device=target).view(1,-1,1,1);std=torch.as_tensor(payload["latent_std"],device=target).view(1,-1,1,1);return cls(model,target,payload["report"],mean,std)
    def predict_latent(self,current,*,action:int,control,state):
        current=current.to(self.device);value=(current-self.mean)/self.std;action_array=np.asarray(action);action_tensor=torch.full((len(value),),int(action_array),dtype=torch.long,device=self.device) if action_array.ndim==0 else torch.as_tensor(action_array,dtype=torch.long,device=self.device).reshape(len(value));control=torch.as_tensor(control,dtype=torch.float32,device=self.device).reshape(len(value),4);state=torch.as_tensor(state,dtype=torch.float32,device=self.device).reshape(len(value),64);time=torch.zeros(len(value),device=self.device)
        with torch.inference_mode():residual=self.model(value,time,action_tensor,control,state)
        return current+residual*self.std
