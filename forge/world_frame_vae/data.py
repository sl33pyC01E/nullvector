from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
from ..action_teacher_v1 import validate_trajectory

def load_episodes(paths):
    frames=[];sources=[]
    for path in map(Path,paths):
        manifest=validate_trajectory(path)
        with np.load(path/manifest["artifact"]["path"],allow_pickle=False) as archive:frames.append(archive["frame"].copy())
        sources.append({"session_id":manifest["session_id"],"manifest_sha256":manifest["manifest_sha256"],"arrays_sha256":manifest["arrays_sha256"],"frames":manifest["frames"]})
    if not frames:raise ValueError("world frame VAE corpus is empty")
    array=np.concatenate(frames).astype(np.uint8);digest=hashlib.sha256(b"nullvector-world-frame-corpus-v1\0")
    for source in sources:digest.update(source["manifest_sha256"].encode()+b"\0"+source["arrays_sha256"].encode()+b"\0")
    digest.update(array.tobytes());return array,tuple(sources),digest.hexdigest()
