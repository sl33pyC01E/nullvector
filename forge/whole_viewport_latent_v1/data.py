from __future__ import annotations

from pathlib import Path
import numpy as np
from ..action_teacher_viewport_v5 import validate_trajectory

NAMES=("frame","spatial","organisms","organism_mask","state","actor_state","actor_field","visibility","memory","control","action")
MACRO_HOLDOUT_INDICES=(0,5,7,14,21,28)

def load_corpus(root:Path):
    episodes=[];manifests=[]
    for manifest_path in sorted(Path(root).glob("*/manifest.json")):
        manifest=validate_trajectory(manifest_path.parent)
        with np.load(manifest_path.parent/manifest["artifact"]["path"],allow_pickle=False) as archive:episode={name:archive[name] for name in NAMES}
        episodes.append(episode);manifests.append(manifest)
    if not episodes:raise ValueError("whole-viewport corpus is empty")
    return episodes,manifests

def rows(episodes):
    result={name:np.concatenate([episode[name] for episode in episodes]) for name in NAMES}
    result["previous_frame"]=np.concatenate([np.concatenate((episode["frame"][:1],episode["frame"][:-1])) for episode in episodes])
    result["episode_index"]=np.concatenate([np.full(len(episode["frame"]),index,np.int32) for index,episode in enumerate(episodes)])
    result["episode_step_index"]=np.concatenate([np.arange(len(episode["frame"]),dtype=np.int32) for episode in episodes])
    return result

def sequence_starts(data, length:int):
    if length < 1: raise ValueError("whole-viewport rollout length drifted")
    count=len(data["frame"])
    if count < length:return np.empty(0,np.int64)
    starts=np.arange(count-length+1,dtype=np.int64)
    return starts[data["episode_index"][starts] == data["episode_index"][starts+length-1]]

def split_episodes(episodes):
    if len(episodes)==30:
        heldout=set(MACRO_HOLDOUT_INDICES)
        return [item for index,item in enumerate(episodes) if index not in heldout],[item for index,item in enumerate(episodes) if index in heldout],list(MACRO_HOLDOUT_INDICES)
    return episodes[:-1],episodes[-1:],[len(episodes)-1]
