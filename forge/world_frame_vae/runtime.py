from __future__ import annotations
from pathlib import Path
import numpy as np,torch
from .contract import CHECKPOINT_FORMAT,ModelConfig,source_sha256
from .model import WorldFrameVAE

class WorldFrameVAERuntime:
    def __init__(self,model,device,report):self.model=model;self.device=device;self.report=report
    @classmethod
    def from_checkpoint(cls,path:Path,*,device="cuda"):
        target=torch.device(device if device!="cuda" or torch.cuda.is_available() else "cpu");payload=torch.load(Path(path),map_location="cpu",weights_only=False)
        if payload.get("format")!=CHECKPOINT_FORMAT or payload.get("source_sha256")!=source_sha256():raise ValueError("world frame VAE checkpoint provenance drifted")
        model=WorldFrameVAE(ModelConfig(**payload["model_config"]));model.load_state_dict(payload["ema_state"]);model.to(target).eval();return cls(model,target,payload["report"])
    def reconstruct(self,frame:np.ndarray)->np.ndarray:
        if frame.shape!=(256,256,3) or frame.dtype!=np.uint8:raise ValueError("world frame VAE input drifted")
        tensor=torch.from_numpy(frame.copy()).permute(2,0,1)[None].float().div_(255).to(self.device)
        with torch.inference_mode():result,_,_=self.model(tensor,sample=False)
        return np.clip(result[0].float().cpu().permute(1,2,0).numpy()*255,0,255).astype(np.uint8)
