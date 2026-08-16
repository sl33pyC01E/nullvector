from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .contract import CHECKPOINT_FORMAT,ModelConfig,TrainingConfig,source_sha256
from .data import VALIDATION_IDENTITIES,LocomotionCorpus,load_corpus
from .model import NeuralLocomotion25D


def _state_hash(state:dict[str,torch.Tensor])->str:
    digest=hashlib.sha256(b"nullvector-torch-state-v1\0")
    for name in sorted(state):digest.update(name.encode()+b"\0"+state[name].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _batch(corpus:LocomotionCorpus,indices:np.ndarray,start:np.ndarray,window:int,device:torch.device)->dict[str,torch.Tensor]:
    dynamic=np.stack([corpus.dynamic[i,s:s+window] for i,s in zip(indices,start)])
    result={
        "global_static":corpus.global_static[indices],"appendage_meta":corpus.appendage_meta[indices],"appendage_mask":corpus.appendage_mask[indices],
        "muscle_meta":corpus.muscle_meta[indices],"muscle_owner":corpus.muscle_owner[indices],"muscle_mask":corpus.muscle_mask[indices],"dynamic":dynamic,
        "contact":np.stack([corpus.contact[i,s:s+window] for i,s in zip(indices,start)]),"muscle":np.stack([corpus.muscle[i,s:s+window] for i,s in zip(indices,start)]),"velocity":np.stack([corpus.velocity[i,s:s+window] for i,s in zip(indices,start)]),
    }
    return {name:torch.from_numpy(value).to(device) for name,value in result.items()}


def _loss(model:NeuralLocomotion25D,batch:dict[str,torch.Tensor])->tuple[torch.Tensor,dict[str,float]]:
    output=model(batch["global_static"],batch["appendage_meta"],batch["appendage_mask"],batch["muscle_meta"],batch["muscle_owner"].long(),batch["muscle_mask"],batch["dynamic"])
    appendage_mask=batch["appendage_mask"][:,None].expand_as(output.contact_logits)
    muscle_mask=batch["muscle_mask"][:,None].expand_as(output.muscle)
    positives=batch["contact"][appendage_mask].sum();negatives=appendage_mask.sum()-positives
    pos_weight=(negatives/positives.clamp_min(1)).clamp(1,6)
    contact=F.binary_cross_entropy_with_logits(output.contact_logits[appendage_mask],batch["contact"][appendage_mask],pos_weight=pos_weight)
    muscle=F.smooth_l1_loss(output.muscle[muscle_mask],batch["muscle"][muscle_mask])
    velocity=F.smooth_l1_loss(output.velocity,batch["velocity"])
    temporal=F.smooth_l1_loss(output.muscle[:,1:]-output.muscle[:,:-1],batch["muscle"][:,1:]-batch["muscle"][:,:-1])
    total=contact*1.6+muscle*2.2+velocity*.8+temporal*.35
    return total,{"loss":float(total.detach()),"contact":float(contact.detach()),"muscle":float(muscle.detach()),"velocity":float(velocity.detach()),"temporal":float(temporal.detach())}


@torch.inference_mode()
def evaluate_arrays(model:NeuralLocomotion25D,corpus:LocomotionCorpus,device:torch.device)->dict[str,float]:
    selected=np.flatnonzero(np.isin(corpus.identity,VALIDATION_IDENTITIES));outputs=[]
    model.eval()
    for start in range(0,len(selected),4):
        ids=selected[start:start+4]
        tensors={name:torch.from_numpy(getattr(corpus,name)[ids]).to(device) for name in ("global_static","appendage_meta","appendage_mask","muscle_meta","muscle_owner","muscle_mask","dynamic")}
        out=model(tensors["global_static"],tensors["appendage_meta"],tensors["appendage_mask"],tensors["muscle_meta"],tensors["muscle_owner"].long(),tensors["muscle_mask"],tensors["dynamic"])
        outputs.append((torch.sigmoid(out.contact_logits).cpu(),out.muscle.cpu(),out.velocity.cpu()))
    contact=torch.cat([o[0] for o in outputs]);muscle=torch.cat([o[1] for o in outputs]);velocity=torch.cat([o[2] for o in outputs])
    truth_contact=torch.from_numpy(corpus.contact[selected]);truth_muscle=torch.from_numpy(corpus.muscle[selected]);truth_velocity=torch.from_numpy(corpus.velocity[selected])
    am=torch.from_numpy(corpus.appendage_mask[selected])[:,None].expand_as(contact);mm=torch.from_numpy(corpus.muscle_mask[selected])[:,None].expand_as(muscle)
    hard=contact>=.5;truth=truth_contact>=.5;tp=int((hard&truth&am).sum());fp=int((hard&~truth&am).sum());fn=int((~hard&truth&am).sum())
    return {"contact_f1":2*tp/max(1,2*tp+fp+fn),"contact_iou":tp/max(1,tp+fp+fn),"muscle_mae":float(torch.abs(muscle[mm]-truth_muscle[mm]).mean()),"velocity_mae":float(torch.abs(velocity-truth_velocity).mean())}


def train(corpus_path:Path,output:Path,*,training:TrainingConfig=TrainingConfig(),model_config:ModelConfig=ModelConfig(),device:str="cuda")->dict[str,Any]:
    corpus=load_corpus(corpus_path);target=torch.device(device)
    if target.type=="cuda" and not torch.cuda.is_available():raise RuntimeError("CUDA is required for 2.5D neural training")
    torch.manual_seed(training.seed);np_rng=np.random.default_rng(training.seed)
    model=NeuralLocomotion25D(model_config).to(target);ema=copy.deepcopy(model).eval();optimizer=torch.optim.AdamW(model.parameters(),lr=training.learning_rate,weight_decay=training.weight_decay,fused=target.type=="cuda")
    scaler=None;history=[];train_indices=np.flatnonzero(~np.isin(corpus.identity,VALIDATION_IDENTITIES));window=30
    model.train()
    for update in range(1,training.updates+1):
        ids=np_rng.choice(train_indices,size=training.batch_size,replace=True);starts=np_rng.integers(0,corpus.dynamic.shape[1]-window+1,size=training.batch_size)
        batch=_batch(corpus,ids,starts,window,target);optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=target.type,dtype=torch.bfloat16,enabled=target.type=="cuda"):
            loss,metrics=_loss(model,batch)
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
        with torch.no_grad():
            for ep,p in zip(ema.parameters(),model.parameters()):ep.lerp_(p,1-training.ema_decay)
        if update==1 or update%100==0 or update==training.updates:
            history.append({"update":update,**{k:round(v,8) for k,v in metrics.items()}})
    validation=evaluate_arrays(ema,corpus,target);model_state={k:v.detach().cpu() for k,v in model.state_dict().items()};ema_state={k:v.detach().cpu() for k,v in ema.state_dict().items()}
    payload={"format":CHECKPOINT_FORMAT,"source_sha256":source_sha256(),"corpus_sha256":corpus.semantic_sha256,"model_config":asdict(model_config),"training_config":asdict(training),"updates":training.updates,"model":model_state,"ema":ema_state,"model_state_sha256":_state_hash(model_state),"ema_state_sha256":_state_hash(ema_state),"validation":validation,"history":history}
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);stage=output.with_suffix(output.suffix+f".tmp-{os.getpid()}");torch.save(payload,stage);os.replace(stage,output)
    report={"format":"nullvector-neural-locomotion-2.5d-training/1.0.0","checkpoint":output.name,"checkpoint_sha256":hashlib.sha256(output.read_bytes()).hexdigest(),"source_sha256":payload["source_sha256"],"corpus_sha256":payload["corpus_sha256"],"parameters":model.parameter_count,"updates":training.updates,"validation":validation,"history":history,"model_state_sha256":payload["model_state_sha256"],"ema_state_sha256":payload["ema_state_sha256"]}
    output.with_suffix(".json").write_text(json.dumps(report,sort_keys=True,indent=2)+"\n","utf-8")
    return report


def load_model(path:Path,*,device:str="cpu",ema:bool=True)->tuple[NeuralLocomotion25D,dict[str,Any]]:
    payload=torch.load(path,map_location="cpu",weights_only=True)
    if payload.get("format")!=CHECKPOINT_FORMAT or payload.get("source_sha256")!=source_sha256():raise ValueError("2.5D neural checkpoint provenance drifted")
    config=ModelConfig(**payload["model_config"]);model=NeuralLocomotion25D(config);state=payload["ema" if ema else "model"]
    if _state_hash(state)!=payload["ema_state_sha256" if ema else "model_state_sha256"]:raise ValueError("2.5D neural checkpoint state drifted")
    model.load_state_dict(state);model.to(device).eval();return model,payload
