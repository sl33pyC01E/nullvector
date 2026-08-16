from __future__ import annotations
from pathlib import Path
import numpy as np,torch
from .contract import CHECKPOINT_FORMAT,ModelConfig,source_sha256
from .model import ActionDiT

class WorldActionDiTRuntime:
    def __init__(self,model,device,report,mean,std):self.model=model;self.device=device;self.report=report;self.mean=mean;self.std=std
    @classmethod
    def from_checkpoint(cls,path:Path,*,device="cuda"):
        target=torch.device(device if device!="cuda" or torch.cuda.is_available() else "cpu");payload=torch.load(Path(path),map_location="cpu",weights_only=False)
        if payload.get("format")!=CHECKPOINT_FORMAT or payload.get("source_sha256")!=source_sha256():raise ValueError("world action DiT checkpoint provenance drifted")
        model=ActionDiT(ModelConfig(**payload["model_config"]));model.load_state_dict(payload["ema_state"]);model.to(target).eval();mean=torch.as_tensor(payload["latent_mean"],device=target).view(1,-1,1,1);std=torch.as_tensor(payload["latent_std"],device=target).view(1,-1,1,1);return cls(model,target,payload["report"],mean,std)
    def predict_latent(self,current,*,action:int,control,state,steps:int=8):
        value=(current.to(self.device)-self.mean)/self.std;action_array=np.asarray(action);action_tensor=torch.full((len(value),),int(action_array),dtype=torch.long,device=self.device) if action_array.ndim==0 else torch.as_tensor(action_array,dtype=torch.long,device=self.device).reshape(len(value));control=torch.as_tensor(control,dtype=torch.float32,device=self.device).reshape(len(value),4);state=torch.as_tensor(state,dtype=torch.float32,device=self.device).reshape(len(value),64)
        with torch.inference_mode():
            for index in range(steps):
                t=torch.full((len(value),),(index+.5)/steps,device=self.device);value=value+self.model(value,t,action_tensor,control,state)/steps
        return value*self.std+self.mean
