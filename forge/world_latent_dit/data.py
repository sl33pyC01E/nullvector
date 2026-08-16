from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np,torch
from ..action_teacher_v1 import validate_trajectory

def encode_episodes(paths,vae_runtime,*,horizon:int=4):
    episodes=[];sources=[]
    for path in map(Path,paths):
        manifest=validate_trajectory(path)
        with np.load(path/manifest["artifact"]["path"],allow_pickle=False) as archive:raw={name:archive[name].copy() for name in ("frame","control","action","state","tick")}
        tensors=[]
        with torch.inference_mode():
            for start in range(0,len(raw["frame"]),8):
                frame=torch.from_numpy(raw["frame"][start:start+8]).permute(0,3,1,2).float().div_(255).to(vae_runtime.device);mean,_=vae_runtime.model.encode(frame);tensors.append(mean.float().cpu().numpy())
        latent=np.concatenate(tensors);count=len(latent)-horizon
        if count<=0:raise ValueError("action DiT episode is shorter than horizon")
        episodes.append({"current":latent[:count],"target":latent[horizon:],"control":raw["control"][:count],"action":raw["action"][:count],"state":raw["state"][:count],"current_frame":raw["frame"][:count],"target_frame":raw["frame"][horizon:],"tick":raw["tick"][:count]})
        sources.append({"session_id":manifest["session_id"],"manifest_sha256":manifest["manifest_sha256"],"arrays_sha256":manifest["arrays_sha256"],"pairs":count})
    digest=hashlib.sha256(b"nullvector-world-action-latents-v1\0")
    for source in sources:digest.update(source["manifest_sha256"].encode()+b"\0"+source["arrays_sha256"].encode()+b"\0")
    for episode in episodes:digest.update(episode["current"].tobytes()+episode["target"].tobytes())
    return tuple(episodes),tuple(sources),digest.hexdigest()
