from __future__ import annotations
import json
from pathlib import Path
import numpy as np,torch
from ..world_latent_dit.contract import ModelConfig
from .contract import CHECKPOINT_FORMAT,DEFAULT_OUTPUT,canonical,file_sha256,source_sha256,state_sha256
from .model import RecurrentActionDiT

class RecurrentActionDiTRuntime:
    def __init__(self,model,device,mean,std,threshold,alpha,report):self.model=model.eval();self.device=device;self.mean=mean;self.std=std;self.threshold=float(threshold);self.alpha=float(alpha);self.report=report
    @classmethod
    def from_output(cls,output:Path=DEFAULT_OUTPUT,*,device="cuda"):
        root=Path(output).resolve();raw=(root/"report.json").read_bytes();report=json.loads(raw)
        if raw!=canonical(report) or report.get("format")!="nullvector-recurrent-action-dit-v2-report/1.0.0" or report.get("source_sha256")!=source_sha256() or report.get("status")!="ready":raise ValueError("recurrent Action-DiT release is not promoted")
        path=root/report["checkpoint"]["path"]
        if file_sha256(path)!=report["checkpoint"]["sha256"]:raise ValueError("recurrent Action-DiT artifact drifted")
        payload=torch.load(path,map_location="cpu",weights_only=True)
        if payload.get("format")!=CHECKPOINT_FORMAT or state_sha256(payload["state"])!=payload.get("state_sha256"):raise ValueError("recurrent Action-DiT state drifted")
        target=torch.device(device if device!="cuda" or torch.cuda.is_available() else "cpu");model=RecurrentActionDiT(ModelConfig(**payload["model_config"]));model.load_state_dict(payload["state"],strict=True);model.to(target)
        mean=torch.tensor(payload["normalization"]["mean"],device=target)[None,:,None,None];std=torch.tensor(payload["normalization"]["std"],device=target)[None,:,None,None]
        return cls(model,target,mean,std,report["selection"]["threshold"],report["selection"]["alpha"],report)
    @torch.inference_mode()
    def step(self,current,previous,*,action,control,state,actor_state):
        current=torch.as_tensor(current,dtype=torch.float32,device=self.device);previous=torch.as_tensor(previous,dtype=torch.float32,device=self.device);cn=(current-self.mean)/self.std;pn=(previous-self.mean)/self.std;action=torch.as_tensor(action,dtype=torch.long,device=self.device).reshape(len(cn));control=torch.as_tensor(control,dtype=torch.float32,device=self.device).reshape(len(cn),4);state=torch.as_tensor(state,dtype=torch.float32,device=self.device).reshape(len(cn),64);actor_state=torch.as_tensor(actor_state,dtype=torch.float32,device=self.device).reshape(len(cn),128);delta=self.model(cn,pn,action,control,state,actor_state);gate=delta.abs().mean(1,keepdim=True)>=self.threshold;return (cn+self.alpha*gate*delta)*self.std+self.mean
