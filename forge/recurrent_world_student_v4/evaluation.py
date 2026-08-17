from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from ..recurrent_world_student_v3.model import RecurrentWorldStudent
from ..safety import require_disk_floor
from ..world_action_clean_v9 import load
from ..world_frame_decoder_adapt_v1 import AdaptedWorldFrameCodec
from ..world_latent_dit.contract import ModelConfig
from .contract import CHECKPOINT_FORMAT, CODEC, CODEC_SHA256, CORPUS, DEFAULT_OUTPUT, FORMAT, PARENT, PARENT_SHA256, REPORT_FORMAT, canonical, file_sha256, source_sha256, state_sha256


def _load(path, expected_sha, device):
    if file_sha256(path)!=expected_sha:raise ValueError("clean recurrent candidate/parent bytes drifted")
    payload=torch.load(path,map_location="cpu",weights_only=True);model=RecurrentWorldStudent(ModelConfig(**payload["model_config"]));model.load_state_dict(payload["state"],strict=True);return model.to(device).eval(),payload


@torch.inference_mode()
def _rollout(model,test,starts,horizon,codec,norms,device):
    lm,ls,am,ass=norms;previous=torch.from_numpy(test["latent"][starts-1]).to(device);current=torch.from_numpy(test["latent"][starts]).to(device);previous_actor=torch.from_numpy(test["actor_state"][starts-1]).to(device);current_actor=torch.from_numpy(test["actor_state"][starts]).to(device);frame=torch.from_numpy(test["frame"][starts]).permute(0,3,1,2).float().div_(255);initial=frame.clone()
    for step in range(horizon):
        indices=starts+step+1;action=torch.from_numpy(test["action"][indices].astype(np.int64)).to(device);control=torch.from_numpy(test["control"][indices]).to(device);state=torch.from_numpy(test["state"][indices-1]).to(device);cn,pn=(current-lm)/ls,(previous-lm)/ls;delta=model.action(cn,pn,action,control,state,current_actor);next_latent=(cn+(delta.abs().mean(1,keepdim=True)>=.18)*delta)*ls+lm;an,pan=(current_actor-am)/ass,(previous_actor-am)/ass;actor=model.actor(an,pan,action,control,state);next_actor=(an+.9*(actor.gate>=.7)*(actor.state-an))*ass+am;frame=torch.clamp(frame+codec.model.decode(next_latent).float().cpu()-codec.model.decode(current).float().cpu(),0,1);previous,current=current,next_latent;previous_actor,current_actor=current_actor,next_actor
    target=torch.from_numpy(test["frame"][starts+horizon]).permute(0,3,1,2).float().div_(255);mae=float(F.l1_loss(frame,target));persistence=float(F.l1_loss(initial,target));return {"samples":len(starts),"mae":mae,"persistence_mae":persistence,"improvement":1-mae/persistence}


def evaluate(output:Path=DEFAULT_OUTPUT):
    output=Path(output).resolve();runtime=output/"runtime.pt";report_path=output/"evaluation.json";require_disk_floor(output.parent,floor_gb=100,planned_bytes=8*1024**2)
    if file_sha256(CODEC)!=CODEC_SHA256:raise ValueError("clean recurrent codec drifted")
    candidate_sha=file_sha256(runtime);device=torch.device("cuda:0");candidate,payload=_load(runtime,candidate_sha,device);parent,parent_payload=_load(PARENT,PARENT_SHA256,device)
    if payload.get("format")!=CHECKPOINT_FORMAT or payload.get("source_sha256")!=source_sha256() or payload.get("parent_sha256")!=PARENT_SHA256 or state_sha256(payload["state"])!=payload.get("state_sha256"):raise ValueError("clean recurrent release drifted")
    codec=AdaptedWorldFrameCodec.from_checkpoint(CODEC,device="cuda");sequences,manifest=load(CORPUS);test=sequences[5];n=payload["normalization"];norms=(torch.tensor(n["latent_mean"],device=device)[None,:,None,None],torch.tensor(n["latent_std"],device=device)[None,:,None,None],torch.tensor(n["actor_mean"],device=device)[None],torch.tensor(n["actor_std"],device=device)[None]);horizons={}
    for horizon in (1,2,4,8,16,32):
        starts=np.linspace(1,len(test["latent"])-horizon-1,min(24,len(test["latent"])-horizon-1),dtype=np.int64);base=_rollout(parent,test,starts,horizon,codec,norms,device);result=_rollout(candidate,test,starts,horizon,codec,norms,device);horizons[str(horizon)]={"parent":base,"candidate":result,"candidate_vs_parent":1-result["mae"]/base["mae"]}
    long=[horizons[str(value)] for value in (4,8,16,32)];gates={"all_long_horizons_beat_persistence":all(row["candidate"]["improvement"]>0 for row in long),"all_long_horizons_beat_parent":all(row["candidate_vs_parent"]>0 for row in long)};gates["all_passed"]=all(gates.values());report={"format":REPORT_FORMAT,"status":"clean_long_horizon_ready" if gates["all_passed"] else "experimental","source_sha256":source_sha256(),"corpus_sha256":manifest["manifest_sha256"],"candidate_sha256":candidate_sha,"parent_sha256":PARENT_SHA256,"codec_sha256":CODEC_SHA256,"horizons":horizons,"gates":gates};report["report_sha256"]=hashlib.sha256(canonical(report)).hexdigest();report_path.write_bytes(canonical(report));return report
