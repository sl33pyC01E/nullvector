from __future__ import annotations

import copy,hashlib,json,os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from ..nature_sim_v2.contract import INTENTS
from .contract import CHECKPOINT_FORMAT,ModelConfig,TrainingConfig,source_sha256
from .corpus import BehaviorCorpus,load_corpus
from .model import NeuralNatureBehavior


def _state_hash(state)->str:
    digest=hashlib.sha256(b"nullvector-nature-behavior-state-v1\0")
    for name in sorted(state):digest.update(name.encode()+b"\0"+state[name].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _tensor_batch(corpus:BehaviorCorpus,indices:np.ndarray,device:torch.device):
    names=("self_features","resource","neighbor","neighbor_mask","intent","direction")
    return {name:torch.from_numpy(getattr(corpus,name)[indices]).to(device) for name in names}


@torch.inference_mode()
def evaluate(model:NeuralNatureBehavior,corpus:BehaviorCorpus,indices:np.ndarray,device:torch.device)->dict[str,object]:
    model.eval();predictions=[];directions=[]
    for start in range(0,len(indices),512):
        batch=_tensor_batch(corpus,indices[start:start+512],device);out=model(batch["self_features"],batch["resource"],batch["neighbor"],batch["neighbor_mask"]);predictions.append(out.intent_logits.argmax(-1).cpu());directions.append(out.direction.float().cpu())
    predicted=torch.cat(predictions).numpy();direction=torch.cat(directions).numpy();truth=corpus.intent[indices];target=corpus.direction[indices]
    per_intent={}
    for index,name in enumerate(INTENTS):
        selected=truth==index
        per_intent[name]={"count":int(selected.sum()),"accuracy":float((predicted[selected]==index).mean()) if selected.any() else None,"predicted":int((predicted==index).sum())}
    active=np.linalg.norm(target,axis=1)>.1;cosine=float(np.mean(np.sum(direction[active]*target[active],axis=1)/(np.linalg.norm(direction[active],axis=1).clip(.01)))) if active.any() else 1
    return {"intent_accuracy":float((predicted==truth).mean()),"direction_mae":float(np.abs(direction-target).mean()),"direction_cosine":cosine,"per_intent":per_intent}


def train(corpus_path:Path,output:Path,*,training:TrainingConfig=TrainingConfig(),model_config:ModelConfig=ModelConfig(),device:str="cuda"):
    corpus=load_corpus(corpus_path);target=torch.device(device)
    if target.type=="cuda" and not torch.cuda.is_available():raise RuntimeError("CUDA required for behavior training")
    torch.manual_seed(training.seed);rng=np.random.default_rng(training.seed);model=NeuralNatureBehavior(model_config).to(target);ema=copy.deepcopy(model).eval();optimizer=torch.optim.AdamW(model.parameters(),lr=training.learning_rate,weight_decay=training.weight_decay,fused=target.type=="cuda")
    validation=np.flatnonzero(corpus.world_id%5==0);train_ids=np.flatnonzero(corpus.world_id%5!=0);counts=np.bincount(corpus.intent[train_ids],minlength=len(INTENTS));weights=np.zeros(len(INTENTS),np.float32);present=counts>0;weights[present]=np.sqrt(counts[present].sum()/counts[present]);weights[present]/=weights[present].mean();class_weights=torch.from_numpy(weights).to(target);history=[]
    model.train()
    for update in range(1,training.updates+1):
        indices=rng.choice(train_ids,training.batch_size,replace=True);batch=_tensor_batch(corpus,indices,target);optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=target.type,dtype=torch.bfloat16,enabled=target.type=="cuda"):
            out=model(batch["self_features"],batch["resource"],batch["neighbor"],batch["neighbor_mask"]);intent=F.cross_entropy(out.intent_logits,batch["intent"].long(),weight=class_weights);direction=F.smooth_l1_loss(out.direction,batch["direction"]);active=torch.linalg.vector_norm(batch["direction"],dim=-1)>.1;angular=(1-F.cosine_similarity(out.direction[active],batch["direction"][active],dim=-1)).mean() if active.any() else direction*0;urgency_target=active.float();urgency=F.binary_cross_entropy_with_logits(out.urgency,urgency_target);loss=intent+direction*2.0+angular*.9+urgency*.25
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
        with torch.no_grad():
            decay=min(training.ema_decay,update/(update+1))
            for ep,p in zip(ema.parameters(),model.parameters()):ep.mul_(decay).add_(p,alpha=1-decay)
        if update==1 or update%100==0 or update==training.updates:history.append({"update":update,"loss":round(float(loss),7),"intent":round(float(intent),7),"direction":round(float(direction),7),"angular":round(float(angular),7)})
    raw=evaluate(model,corpus,validation,target);ema_metrics=evaluate(ema,corpus,validation,target);score=lambda value:value["intent_accuracy"]+value["direction_cosine"]-value["direction_mae"];selected="ema" if score(ema_metrics)>=score(raw) else "model";model_state={k:v.detach().cpu() for k,v in model.state_dict().items()};ema_state={k:v.detach().cpu() for k,v in ema.state_dict().items()}
    payload={"format":CHECKPOINT_FORMAT,"source_sha256":source_sha256(),"corpus_sha256":corpus.semantic_sha256,"model_config":asdict(model_config),"training_config":asdict(training),"selected":selected,"model":model_state,"ema":ema_state,"model_state_sha256":_state_hash(model_state),"ema_state_sha256":_state_hash(ema_state),"validation":{"model":raw,"ema":ema_metrics,"selected":selected},"history":history}
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);stage=output.with_suffix(output.suffix+f".tmp-{os.getpid()}");torch.save(payload,stage);os.replace(stage,output);report={"format":CHECKPOINT_FORMAT,"checkpoint_sha256":hashlib.sha256(output.read_bytes()).hexdigest(),"parameters":model.parameter_count,"source_sha256":source_sha256(),"corpus_sha256":corpus.semantic_sha256,"validation":payload["validation"],"history":history,"model_state_sha256":payload["model_state_sha256"],"ema_state_sha256":payload["ema_state_sha256"]};output.with_suffix(".json").write_text(json.dumps(report,sort_keys=True,indent=2)+"\n","utf-8");return report


def load_model(path:Path,*,device="cpu"):
    payload=torch.load(path,map_location="cpu",weights_only=True)
    if payload.get("format")!=CHECKPOINT_FORMAT or payload.get("source_sha256")!=source_sha256():raise ValueError("behavior checkpoint provenance drifted")
    state=payload[payload["selected"]];expected=payload[f"{payload['selected']}_state_sha256"]
    if _state_hash(state)!=expected:raise ValueError("behavior checkpoint state drifted")
    model=NeuralNatureBehavior(ModelConfig(**payload["model_config"]));model.load_state_dict(state);return model.to(device).eval(),payload
