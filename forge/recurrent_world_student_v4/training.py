from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from ..recurrent_world_student_v3.model import RecurrentWorldStudent
from ..recurrent_world_student_v3.training import _batch, _one_step_metrics, _tensors
from ..safety import require_disk_floor
from ..world_action_clean_v9 import load
from ..world_latent_dit.contract import ModelConfig
from .contract import CHECKPOINT_FORMAT, CORPUS, DEFAULT_OUTPUT, PARENT, PARENT_SHA256, TrainingPlan, file_sha256, source_sha256, state_sha256


def _atomic(path, payload):
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _normalizers(parent, device):
    values = parent["normalization"]
    return (
        torch.tensor(values["latent_mean"], device=device)[None, :, None, None],
        torch.tensor(values["latent_std"], device=device)[None, :, None, None],
        torch.tensor(values["actor_mean"], device=device)[None],
        torch.tensor(values["actor_std"], device=device)[None],
    )


def train(output: Path = DEFAULT_OUTPUT, *, plan: TrainingPlan = TrainingPlan()):
    output = Path(output).resolve();output.mkdir(parents=True, exist_ok=True);require_disk_floor(output.parent, floor_gb=100, planned_bytes=2 * 1024**3)
    if not torch.cuda.is_available():raise RuntimeError("clean recurrent V4 training requires CUDA")
    if file_sha256(PARENT) != PARENT_SHA256:raise ValueError("clean recurrent V4 parent drifted")
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.48, 0);torch.manual_seed(plan.seed);rng=np.random.default_rng(plan.seed);device=torch.device("cuda:0")
    sequences, manifest = load(CORPUS);parent=torch.load(PARENT,map_location="cpu",weights_only=True);model=RecurrentWorldStudent(ModelConfig(**parent["model_config"]));model.load_state_dict(parent["state"],strict=True);model.to(device);ema=copy.deepcopy(model).eval().requires_grad_(False);optimizer=torch.optim.AdamW(model.parameters(),lr=plan.learning_rate,weight_decay=1e-3,fused=True);norms=_normalizers(parent,device);lm,ls,am,ass=norms;history=[];start_update=0;latest=output/"latest.pt"
    if latest.is_file():
        payload=torch.load(latest,map_location="cpu",weights_only=True)
        if payload.get("format")!=CHECKPOINT_FORMAT or payload.get("source_sha256")!=source_sha256() or payload.get("corpus_sha256")!=manifest["manifest_sha256"] or payload.get("parent_sha256")!=PARENT_SHA256:raise ValueError("clean recurrent V4 resume drifted")
        model.load_state_dict(payload["model_state"]);ema.load_state_dict(payload["ema_state"]);optimizer.load_state_dict(payload["optimizer_state"]);rng.bit_generator.state=payload["rng_state"];history=list(payload["history"]);start_update=payload["update"]
    for end in range(start_update+plan.segment_updates,plan.total_updates+1,plan.segment_updates):
        began=time.perf_counter();torch.cuda.reset_peak_memory_stats(device);model.train()
        for update in range(end-plan.segment_updates+1,end+1):
            rows=_batch(sequences[:4],rng,plan.batch_size,plan.rollout_steps);optimizer.zero_grad(set_to_none=True);latent_total=actor_total=0.;previous=current=previous_actor=actor=None
            for offset in range(plan.rollout_steps):
                values=_tensors(rows,offset,device)
                if offset==0:previous,current=values["previous"],values["current"];previous_actor,actor=values["previous_actor"],values["actor"]
                target,target_actor=values["target"],values["target_actor"];cn,pn=(current-lm)/ls,(previous-lm)/ls;an,pan,tan=(actor-am)/ass,(previous_actor-am)/ass,(target_actor-am)/ass
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    delta=model.action(cn,pn,values["action"],values["control"],values["state"],actor);magnitude=((target-current)/ls).abs().mean(1,keepdim=True);latent_loss=(F.smooth_l1_loss(delta,(target-current)/ls,reduction="none")*(1+5*torch.clamp(magnitude/.35,0,2))).mean();actor_result=model.actor(an,pan,values["action"],values["control"],values["state"]);changed=(tan-an).abs()>.025;actor_loss=(F.smooth_l1_loss(actor_result.state,tan,reduction="none")*(1+6*changed)).mean();loss=(latent_loss+plan.actor_weight*actor_loss)/plan.rollout_steps
                loss.backward();latent_total+=float(latent_loss);actor_total+=float(actor_loss)
                with torch.no_grad():
                    next_latent=(cn+(delta.abs().mean(1,keepdim=True)>=.18)*delta)*ls+lm;next_actor=(an+.9*(actor_result.gate>=.7)*(actor_result.state-an))*ass+am
                previous,current=current.detach(),next_latent.detach();previous_actor,actor=actor.detach(),next_actor.detach()
            gradient=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1));optimizer.step()
            with torch.no_grad():torch._foreach_mul_(list(ema.parameters()),plan.ema_decay);torch._foreach_add_(list(ema.parameters()),list(model.parameters()),alpha=1-plan.ema_decay)
            if update==1 or update%25==0:history.append({"update":update,"latent":round(latent_total/plan.rollout_steps,7),"actor":round(actor_total/plan.rollout_steps,7),"gradient":round(gradient,7)})
        raw_validation=_one_step_metrics(model.eval(),sequences[4],norms,device);ema_validation=_one_step_metrics(ema.eval(),sequences[4],norms,device);model_state={name:value.detach().cpu() for name,value in model.state_dict().items()};ema_state={name:value.detach().cpu() for name,value in ema.state_dict().items()};payload={"format":CHECKPOINT_FORMAT,"source_sha256":source_sha256(),"corpus_sha256":manifest["manifest_sha256"],"parent_sha256":PARENT_SHA256,"model_config":parent["model_config"],"plan":plan.to_dict(),"update":end,"model_state":model_state,"ema_state":ema_state,"optimizer_state":optimizer.state_dict(),"rng_state":rng.bit_generator.state,"history":history,"normalization":parent["normalization"],"validation":{"raw":raw_validation,"ema":ema_validation},"runtime":{"segment_seconds":round(time.perf_counter()-began,6),"peak_reserved_bytes":int(torch.cuda.max_memory_reserved(device))}};_atomic(latest,payload);_atomic(output/f"milestone_{end:07d}.pt",payload);print(json.dumps({"update":end,"raw":raw_validation["improvement"],"ema":ema_validation["improvement"],**payload["runtime"]}),flush=True)
    variant="raw" if payload["validation"]["raw"]["improvement"]>=payload["validation"]["ema"]["improvement"] else "ema";chosen=model if variant=="raw" else ema;state={name:value.detach().cpu() for name,value in chosen.state_dict().items()};release={"format":CHECKPOINT_FORMAT,"source_sha256":source_sha256(),"corpus_sha256":manifest["manifest_sha256"],"parent_sha256":PARENT_SHA256,"model_config":parent["model_config"],"state":state,"state_sha256":state_sha256(state),"normalization":parent["normalization"],"selection":{"variant":variant,"validation":payload["validation"][variant]},"plan":plan.to_dict()};_atomic(output/"runtime.pt",release);return {"status":"trained_pending_long_horizon_evaluation","updates":plan.total_updates,"variant":variant,"validation":release["selection"]["validation"],"checkpoint_sha256":file_sha256(output/"runtime.pt"),"state_sha256":release["state_sha256"]}
