from __future__ import annotations
from pathlib import Path
import hashlib,numpy as np,torch
from ..world_frame_vae import WorldFrameVAERuntime
from .contract import CHECKPOINT_FORMAT,ModelConfig,source_sha256
from .model import PixelCellRefiner

def file_sha256(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1<<20),b""):digest.update(chunk)
    return digest.hexdigest()

class RefinedWorldFrameVAERuntime:
    def __init__(self,base,refiner,device,report):self.base=base;self.model=base.model;self.refiner=refiner;self.device=device;self.report=report
    @classmethod
    def from_checkpoints(cls,base_path:Path,refiner_path:Path,*,device="cuda"):
        target=torch.device(device if device!="cuda" or torch.cuda.is_available() else "cpu");payload=torch.load(Path(refiner_path),map_location="cpu",weights_only=False)
        if payload.get("format")!=CHECKPOINT_FORMAT or payload.get("source_sha256")!=source_sha256():raise ValueError("world VAE pixel refiner provenance drifted")
        if payload.get("base_checkpoint_sha256")!=file_sha256(base_path):raise ValueError("world VAE pixel refiner base checkpoint drifted")
        base=WorldFrameVAERuntime.from_checkpoint(base_path,device=str(target));refiner=PixelCellRefiner(ModelConfig(**payload["model_config"]));refiner.load_state_dict(payload["ema_state"]);refiner.to(target).eval();return cls(base,refiner,target,payload["report"])
    def encode(self,frame):
        value=torch.as_tensor(np.asarray(frame),device=self.device).permute(2,0,1).unsqueeze(0).float()/255;return self.model.encode(value)[0]
    def decode(self,latent):
        with torch.inference_mode():return self.refiner(self.model.decode(latent.to(self.device)))
    def reconstruct(self,frame):return np.clip(self.decode(self.encode(frame))[0].permute(1,2,0).float().cpu().numpy()*255,0,255).astype(np.uint8)
