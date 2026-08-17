from __future__ import annotations
import json
from pathlib import Path
import numpy as np,torch
from ..actor_state_student_v1.model import ActorStateStudent
from ..config import PROJECT_ROOT
from ..organism_cell_vae_runtime_v1 import ContinuousCellVAERuntime
from ..living_body_nca_v1 import LivingBodyNCARuntime
from ..world_frame_vae.contract import ModelConfig as VAEConfig
from ..world_frame_vae.model import WorldFrameVAE
from ..world_latent_dit.contract import ModelConfig as DiTConfig
from ..world_latent_dit.model import ActionDiT
from .contract import ARTIFACTS,file_sha256,tensor_state_sha256
def _load(relative):return torch.load(PROJECT_ROOT/relative,map_location="cpu",weights_only=True)
def _bound(name,payload,report_relative,artifact_relative):
    report=json.loads((PROJECT_ROOT/report_relative).read_text("utf-8"))
    if payload.get("source_sha256")!=report.get("source_sha256") or payload.get("report")!=report:raise ValueError(f"composite {name} report binding drifted")
    if tensor_state_sha256(payload["ema_state"])!=payload.get("ema_sha256"):raise ValueError(f"composite {name} state hash drifted")
    return report
class CompositeWorldRuntime:
    """Callable Action-DiT + VAE world path with causal actor/body specialists."""
    def __init__(self,dit,vae,actor,device,dit_mean,dit_std,actor_mean,actor_std,actor_threshold,actor_alpha,organism,physiology):self.dit=dit;self.vae=vae;self.actor=actor;self.device=device;self.dit_mean=dit_mean;self.dit_std=dit_std;self.actor_mean=actor_mean;self.actor_std=actor_std;self.actor_threshold=actor_threshold;self.actor_alpha=actor_alpha;self.organism=organism;self.physiology=physiology
    @classmethod
    def from_release(cls,*,device="cuda"):
        target=torch.device(device if device!="cuda" or torch.cuda.is_available() else "cpu")
        dit_report,dit_art=ARTIFACTS["action_dit"];dit_payload=_load(dit_art);_bound("Action-DiT",dit_payload,dit_report,dit_art);dit=ActionDiT(DiTConfig(**dit_payload["model_config"]));dit.load_state_dict(dit_payload["ema_state"],strict=True);dit.to(target).eval();dit_mean=torch.tensor(dit_payload["latent_mean"],device=target)[None,:,None,None];dit_std=torch.tensor(dit_payload["latent_std"],device=target)[None,:,None,None]
        vae_report,vae_art=ARTIFACTS["world_vae"];vae_payload=_load(vae_art);_bound("world VAE",vae_payload,vae_report,vae_art);vae=WorldFrameVAE(VAEConfig(**vae_payload["model_config"]));vae.load_state_dict(vae_payload["ema_state"],strict=True);vae.to(target).eval()
        actor_report,actor_art=ARTIFACTS["actor_state"];report=json.loads((PROJECT_ROOT/actor_report).read_text("utf-8"));actor_payload=_load(actor_art);key="model_state" if report["selection"]["variant"]=="raw" else "ema_state";actor=ActorStateStudent();actor.load_state_dict(actor_payload[key],strict=True);actor.to(target).eval();actor_mean=torch.tensor(actor_payload["normalization"]["mean"],device=target);actor_std=torch.tensor(actor_payload["normalization"]["std"],device=target)
        organism=ContinuousCellVAERuntime.from_release(device=str(target));physiology=LivingBodyNCARuntime.from_output(device=str(target))
        return cls(dit,vae,actor,target,dit_mean,dit_std,actor_mean,actor_std,report["selection"]["threshold"],report["selection"]["alpha"],organism,physiology)
    @torch.inference_mode()
    def encode(self,frame):
        value=torch.as_tensor(np.asarray(frame),device=self.device).permute(2,0,1).unsqueeze(0).float()/255;return self.vae.encode(value)[0]
    @torch.inference_mode()
    def decode(self,latent):return self.vae.decode(latent.to(self.device))
    @torch.inference_mode()
    def step_visual(self,latent,*,action,control,state,steps=8):
        value=(latent.to(self.device)-self.dit_mean)/self.dit_std;actions=torch.as_tensor(action,dtype=torch.long,device=self.device).reshape(-1);control=torch.as_tensor(control,dtype=torch.float32,device=self.device).reshape(len(value),4);state=torch.as_tensor(state,dtype=torch.float32,device=self.device).reshape(len(value),64)
        for index in range(steps):
            time=torch.full((len(value),),(index+.5)/steps,device=self.device);value=value+self.dit(value,time,actions,control,state)/steps
        return value*self.dit_std+self.dit_mean
    @torch.inference_mode()
    def step_actor(self,current,previous,*,action,control,state):
        current=torch.as_tensor(current,dtype=torch.float32,device=self.device);previous=torch.as_tensor(previous,dtype=torch.float32,device=self.device);cn=(current-self.actor_mean)/self.actor_std;pn=(previous-self.actor_mean)/self.actor_std;action=torch.as_tensor(action,dtype=torch.long,device=self.device).reshape(len(cn));control=torch.as_tensor(control,dtype=torch.float32,device=self.device).reshape(len(cn),4);state=torch.as_tensor(state,dtype=torch.float32,device=self.device).reshape(len(cn),64);result=self.actor(cn,pn,action,control,state);keep=result.gate>=self.actor_threshold;return (cn+self.actor_alpha*keep*(result.state-cn))*self.actor_std+self.actor_mean
