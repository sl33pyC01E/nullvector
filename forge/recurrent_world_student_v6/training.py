from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from ..recurrent_world_student_v5.model import PerceptionRecurrentWorldStudent
from ..safety import require_disk_floor
from ..world_action_natural_v10 import load
from ..world_latent_dit.contract import ModelConfig
from .contract import CHECKPOINT_FORMAT,CORPUS,DEFAULT_OUTPUT,PARENT,PARENT_SHA256,TrainingPlan,file_sha256,source_sha256,state_sha256


def _atomic(path,payload):
    temporary=path.with_name(f".{path.name}.tmp-{os.getpid()}");torch.save(payload,temporary);os.replace(temporary,path)


def _normalizers(payload,device):
    values=payload["normalization"]
    return (torch.tensor(values["latent_mean"],device=device)[None,:,None,None],torch.tensor(values["latent_std"],device=device)[None,:,None,None],torch.tensor(values["actor_mean"],device=device)[None],torch.tensor(values["actor_std"],device=device)[None])


def _sample_batch(sequences,rng,count,steps,device):
    rows=[]
    for _ in range(count):
        sequence=sequences[int(rng.integers(0,len(sequences)))];start=int(rng.integers(1,len(sequence["latent"])-steps));rows.append((sequence,start))
    def gather(name,begin,length):
        array=np.stack([sequence[name][start+begin:start+begin+length] for sequence,start in rows]);return torch.from_numpy(array).to(device,non_blocking=True)
    return {"latent":gather("latent",-1,steps+2),"actor":gather("actor_state",-1,steps+2),"action":gather("action",1,steps).long(),"control":gather("control",1,steps),"state":gather("state",0,steps),"visibility":gather("visibility",0,steps),"memory":gather("memory",0,steps)}


@torch.inference_mode()
def _rollout_metrics(model,sequence,norms,device,horizon=4,perception="normal",samples=32):
    lm,ls,am,ass=norms;starts=np.linspace(1,len(sequence["latent"])-horizon-1,min(samples,len(sequence["latent"])-horizon-1),dtype=np.int64);previous=torch.from_numpy(sequence["latent"][starts-1]).to(device);current=torch.from_numpy(sequence["latent"][starts]).to(device);previous_actor=torch.from_numpy(sequence["actor_state"][starts-1]).to(device);actor=torch.from_numpy(sequence["actor_state"][starts]).to(device)
    for offset in range(horizon):
        indices=starts+offset;action=torch.from_numpy(sequence["action"][indices+1].astype(np.int64)).to(device);control=torch.from_numpy(sequence["control"][indices+1]).to(device);state=torch.from_numpy(sequence["state"][indices]).to(device);visibility=torch.from_numpy(sequence["visibility"][indices]).to(device);memory=torch.from_numpy(sequence["memory"][indices]).to(device)
        if perception=="zero":visibility=torch.zeros_like(visibility);memory=torch.zeros_like(memory)
        elif perception=="shuffle":visibility=visibility.flip(0);memory=memory.flip(0)
        cn,pn=(current-lm)/ls,(previous-lm)/ls;delta,logits=model.gated_action(cn,pn,action,control,state,actor,visibility,memory);next_latent=(cn+torch.sigmoid(logits)*delta)*ls+lm;an,pan=(actor-am)/ass,(previous_actor-am)/ass;actor_result=model.actor(an,pan,action,control,state,visibility,memory);next_actor=(an+.9*(actor_result.gate>=.7)*(actor_result.state-an))*ass+am;previous,current=current,next_latent;previous_actor,actor=actor,next_actor
    target=torch.from_numpy(sequence["latent"][starts+horizon]).to(device);initial=torch.from_numpy(sequence["latent"][starts]).to(device);mae=float(F.l1_loss(current,target));persistence=float(F.l1_loss(initial,target));return {"horizon":horizon,"samples":len(starts),"mae":mae,"persistence_mae":persistence,"improvement":1-mae/persistence}


def train(output:Path=DEFAULT_OUTPUT,*,plan:TrainingPlan=TrainingPlan()):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True);require_disk_floor(output.parent,floor_gb=100,planned_bytes=3*1024**3)
    if not torch.cuda.is_available():raise RuntimeError("recurrent rollout V6 training requires CUDA")
    if file_sha256(PARENT)!=PARENT_SHA256:raise ValueError("recurrent rollout V6 parent drifted")
    if plan.total_updates%plan.segment_updates:raise ValueError("total_updates must be divisible by segment_updates")
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.45,0);torch.manual_seed(plan.seed);rng=np.random.default_rng(plan.seed);device=torch.device("cuda:0");sequences,manifest=load(CORPUS);parent=torch.load(PARENT,map_location="cpu",weights_only=True);model=PerceptionRecurrentWorldStudent(ModelConfig(**parent["model_config"]));model.load_state_dict(parent["state"],strict=True);model.to(device);ema=copy.deepcopy(model).eval().requires_grad_(False);optimizer=torch.optim.AdamW(model.parameters(),lr=plan.learning_rate,weight_decay=1e-3,fused=True);norms=_normalizers(parent,device);lm,ls,am,ass=norms;history=[];start_update=0;latest=output/"latest.pt"
    if latest.is_file():
        payload=torch.load(latest,map_location="cpu",weights_only=True)
        if payload.get("format")!=CHECKPOINT_FORMAT or payload.get("source_sha256")!=source_sha256() or payload.get("corpus_sha256")!=manifest["manifest_sha256"] or payload.get("parent_sha256")!=PARENT_SHA256 or payload.get("plan")!=plan.to_dict():raise ValueError("recurrent rollout V6 resume drifted")
        model.load_state_dict(payload["model_state"]);ema.load_state_dict(payload["ema_state"]);optimizer.load_state_dict(payload["optimizer_state"]);rng.bit_generator.state=payload["rng_state"];history=list(payload["history"]);start_update=payload["update"]
    payload=None
    for end in range(start_update+plan.segment_updates,plan.total_updates+1,plan.segment_updates):
        began=time.perf_counter();torch.cuda.reset_peak_memory_stats(device);model.train()
        for update in range(end-plan.segment_updates+1,end+1):
            batch=_sample_batch(sequences[:4],rng,plan.batch_size,plan.rollout_steps,device);optimizer.zero_grad(set_to_none=True);previous=batch["latent"][:,0];current=batch["latent"][:,1];previous_actor=batch["actor"][:,0];actor=batch["actor"][:,1];latent_total=actor_total=0.
            for offset in range(plan.rollout_steps):
                target=batch["latent"][:,offset+2];target_actor=batch["actor"][:,offset+2];visibility=batch["visibility"][:,offset];memory=batch["memory"][:,offset]
                if plan.perception_dropout and float(rng.random())<plan.perception_dropout:visibility=torch.zeros_like(visibility);memory=torch.zeros_like(memory)
                cn,pn=(current-lm)/ls,(previous-lm)/ls;an,pan,tan=(actor-am)/ass,(previous_actor-am)/ass,(target_actor-am)/ass
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    delta,logits=model.gated_action(cn,pn,batch["action"][:,offset],batch["control"][:,offset],batch["state"][:,offset],actor,visibility,memory);target_delta=(target-current)/ls;magnitude=target_delta.abs().mean(1,keepdim=True)
                    with torch.no_grad():proposal=delta.float();truth=target_delta.float();trust=torch.clamp((proposal*truth).sum(1,keepdim=True)/(proposal.square().sum(1,keepdim=True)+1e-6),0,1)
                    gated=torch.sigmoid(logits)*delta;weight=1+5*torch.clamp(magnitude/.35,0,2);transition=(F.smooth_l1_loss(gated,target_delta,reduction="none")*weight).mean();proposal_loss=(F.smooth_l1_loss(delta,target_delta,reduction="none")*weight).mean();gate_loss=F.smooth_l1_loss(torch.sigmoid(logits),trust);actor_result=model.actor(an,pan,batch["action"][:,offset],batch["control"][:,offset],batch["state"][:,offset],visibility,memory);changed=(tan-an).abs()>.025;actor_loss=(F.smooth_l1_loss(actor_result.state,tan,reduction="none")*(1+6*changed)).mean();latent_loss=transition+plan.proposal_weight*proposal_loss+plan.gate_weight*gate_loss;loss=(latent_loss+plan.actor_weight*actor_loss)/plan.rollout_steps
                loss.backward();latent_total+=float(latent_loss);actor_total+=float(actor_loss)
                with torch.no_grad():next_latent=(cn+torch.sigmoid(logits)*delta)*ls+lm;next_actor=(an+.9*(actor_result.gate>=.7)*(actor_result.state-an))*ass+am
                previous,current=current.detach(),next_latent.detach();previous_actor,actor=actor.detach(),next_actor.detach()
            gradient=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1));optimizer.step()
            with torch.no_grad():torch._foreach_mul_(list(ema.parameters()),plan.ema_decay);torch._foreach_add_(list(ema.parameters()),list(model.parameters()),alpha=1-plan.ema_decay)
            if update%10==0:history.append({"update":update,"latent":round(latent_total/plan.rollout_steps,7),"actor":round(actor_total/plan.rollout_steps,7),"gradient":round(gradient,7)})
        validation={str(h):_rollout_metrics(ema.eval(),sequences[4],norms,device,h) for h in (1,2,4,8)};score=sum(validation[str(h)]["mae"]/validation[str(h)]["persistence_mae"] for h in (1,2,4,8));model_state={n:v.detach().cpu() for n,v in model.state_dict().items()};ema_state={n:v.detach().cpu() for n,v in ema.state_dict().items()};elapsed=time.perf_counter()-began;payload={"format":CHECKPOINT_FORMAT,"source_sha256":source_sha256(),"corpus_sha256":manifest["manifest_sha256"],"parent_sha256":PARENT_SHA256,"model_config":parent["model_config"],"normalization":parent["normalization"],"plan":plan.to_dict(),"update":end,"model_state":model_state,"ema_state":ema_state,"optimizer_state":optimizer.state_dict(),"rng_state":rng.bit_generator.state,"history":history,"validation":validation,"selection_score":score,"runtime":{"segment_seconds":round(elapsed,6),"updates_per_second":round(plan.segment_updates/elapsed,4),"peak_reserved_bytes":int(torch.cuda.max_memory_reserved(device))}};_atomic(latest,payload);_atomic(output/f"milestone_{end:07d}.pt",payload);print(json.dumps({"update":end,"validation":validation,"selection_score":score,"runtime":payload["runtime"]}),flush=True)
    return {"status":"trained_pending_long_horizon_selection","update":payload["update"],"selection_score":payload["selection_score"],"runtime":payload["runtime"],"latest_sha256":file_sha256(latest)}
