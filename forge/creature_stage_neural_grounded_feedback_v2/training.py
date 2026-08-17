from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import time
import uuid

import numpy as np
import torch
import torch.nn.functional as F

from ..creature_stage_neural_motion.training import _state_sha256
from ..safety import require_disk_floor
from .contract import CHECKPOINT_FORMAT, FORMAT, ModelConfig, TrainingConfig, source_sha256
from .dataset import FeedbackCorpus, build_corpus
from .model import NeuralGroundedFeedback
from .physics import simulate_feedback_cycle
from .runtime import NeuralGroundedFeedbackRuntime


def _loss(model: NeuralGroundedFeedback, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    output = model(batch["owner_state"], batch["global_state"], batch["owner_mask"],
                   batch["muscle_meta"], batch["muscle_owner"], batch["muscle_mask"])
    muscle = F.smooth_l1_loss(output.muscle_activation[batch["muscle_mask"]],
                              batch["muscle_target"].float()[batch["muscle_mask"]], beta=.04)
    logits = output.contact_logits[batch["owner_mask"]]
    contact_target = batch["contact_target"].float()[batch["owner_mask"]]
    positive = contact_target.sum().clamp_min(1); negative = (1-contact_target).sum().clamp_min(1)
    contact = F.binary_cross_entropy_with_logits(logits, contact_target, pos_weight=(negative/positive).clamp(.5, 4))
    body = F.smooth_l1_loss(output.body_velocity, batch["body_target"].float(), beta=.03)
    total = muscle * 2.0 + contact * .60 + body * 1.25
    if not bool(torch.isfinite(total)): raise FloatingPointError("grounded feedback loss became non-finite")
    return total, {"loss": float(total.detach()), "muscle": float(muscle.detach()),
                   "contact": float(contact.detach()), "body": float(body.detach())}


@torch.inference_mode()
def _offline(model: NeuralGroundedFeedback, corpus: FeedbackCorpus, device: torch.device) -> dict[str, float]:
    muscle_errors=[]; body_errors=[]; tp=fp=fn=0
    for start in range(0, corpus.samples, 1024):
        indices=torch.arange(start, min(start+1024, corpus.samples)); batch=corpus.batch(indices, device)
        out=model(batch["owner_state"],batch["global_state"],batch["owner_mask"],batch["muscle_meta"],batch["muscle_owner"],batch["muscle_mask"])
        muscle_errors.append((out.muscle_activation-batch["muscle_target"].float()).abs()[batch["muscle_mask"]].cpu())
        body_errors.append((out.body_velocity-batch["body_target"].float()).abs().cpu())
        pred=(torch.sigmoid(out.contact_logits)>=.5)&batch["owner_mask"]; truth=(batch["contact_target"]>.5)&batch["owner_mask"]
        tp+=int((pred&truth).sum()); fp+=int((pred&~truth).sum()); fn+=int((~pred&truth).sum())
    muscle=torch.cat(muscle_errors); body=torch.cat(body_errors)
    return {"muscle_mae":float(muscle.mean()),"muscle_p95":float(torch.quantile(muscle,.95)),
            "body_drive_mae":float(body.mean()),"contact_f1":2*tp/max(2*tp+fp+fn,1),
            "contact_iou":tp/max(tp+fp+fn,1)}


class _ZeroPolicy:
    def predict(self, organism, nodes_local, node_velocity, previous_contact, phase, body_velocity):
        return np.zeros(len(organism.muscles),np.float32),np.zeros(len(organism.genome.appendages),np.bool_),0.0


@torch.inference_mode()
def _rollout(model: NeuralGroundedFeedback, corpus: FeedbackCorpus, device: torch.device) -> dict[str,float]:
    runtime=NeuralGroundedFeedbackRuntime(model,device); distance=[]; ratios=[]; node=[]
    slip=strain=vertical=seam=0.0
    # Five balanced held-out sentinels keep the expensive causal gate bounded.
    for family in range(5):
        index=next(i for i,o in enumerate(corpus.organisms) if int(np.argmax(o.genome.family_mix))==family)
        organism=corpus.organisms[index]; teacher=corpus.cycles[index]
        cycle=simulate_feedback_cycle(organism,runtime)
        distance.append(cycle.distance_px); ratios.append(cycle.distance_px/max(teacher.distance_px,1e-5))
        node.append(float(np.mean([np.abs(a.nodes_local-b.nodes_local).mean() for a,b in zip(cycle.frames,teacher.frames,strict=True)])))
        slip=max(slip,cycle.maximum_contact_slip_px); strain=max(strain,cycle.maximum_edge_strain)
        vertical=max(vertical,cycle.vertical_axis_max_degrees); seam=max(seam,cycle.loop_seam_max_abs)
    zero=[]
    # Native anomaly levitation is an environmental field, not neural drive;
    # score contact-controller ablation only on contact-bearing phenotypes.
    for family in (0,1,2,4):
        index=next(i for i,o in enumerate(corpus.organisms) if int(np.argmax(o.genome.family_mix))==family)
        zero.append(abs(simulate_feedback_cycle(corpus.organisms[index],_ZeroPolicy()).distance_px))
    return {"rollout_distance_mean":float(np.mean(distance)),"advance_ratio_min":float(np.min(ratios)),
            "advance_ratio_max":float(np.max(ratios)),"rollout_node_l1":float(np.mean(node)),
            "maximum_contact_slip_px":slip,"maximum_edge_strain":strain,
            "vertical_axis_max_degrees":vertical,"loop_seam_max_abs":seam,
            "zero_policy_distance_mean":float(np.mean(zero)),
            "causal_distance_gain":float(np.mean(np.abs(distance))/(np.mean(zero)+1e-5))}


def _gates(m:dict[str,float])->dict[str,bool]:
    gates={"muscle_accuracy":m["muscle_mae"]<=.075 and m["muscle_p95"]<=.22,
           "contact_accuracy":m["contact_f1"]>=.90 and m["contact_iou"]>=.82,
           "body_drive_accuracy":m["body_drive_mae"]<=.09,
           "causal_rollout":m["advance_ratio_min"]>=.70 and m["advance_ratio_max"]<=1.35 and m["causal_distance_gain"]>=3,
           "grounded_physics_floor":m["maximum_contact_slip_px"]<.05 and m["maximum_edge_strain"]<.12 and m["vertical_axis_max_degrees"]<5 and m["loop_seam_max_abs"]<.002,
           "closed_loop_shape":m["rollout_node_l1"]<=1.25}
    gates["all_passed"]=all(gates.values()); return gates


def train(output:Path,*,updates:int|None=None,device:str="cuda")->dict[str,object]:
    output=Path(output).resolve()
    if output.exists(): raise FileExistsError(output)
    require_disk_floor(output.parent,floor_gb=100,planned_bytes=2*1024**3)
    defaults=TrainingConfig(); config=TrainingConfig(updates=defaults.updates if updates is None else updates)
    target=torch.device(device if device!="cuda" or torch.cuda.is_available() else "cpu")
    train_corpus=build_corpus(split="train",variants_per_family=config.variants_per_family)
    validation=build_corpus(split="validation",variants_per_family=config.variants_per_family)
    torch.manual_seed(config.seed); np.random.seed(config.seed&0xffffffff)
    if target.type=="cuda": torch.cuda.manual_seed_all(config.seed); torch.cuda.reset_peak_memory_stats(target)
    model_config=ModelConfig(); model=NeuralGroundedFeedback(model_config).to(target).train(); ema=copy.deepcopy(model).eval()
    optimizer=torch.optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay,fused=target.type=="cuda")
    generator=torch.Generator().manual_seed(config.seed); history=[]; started=time.perf_counter()
    for update in range(1,config.updates+1):
        indices=torch.randint(0,train_corpus.samples,(config.batch_size,),generator=generator)
        batch=train_corpus.batch(indices,target); optimizer.zero_grad(set_to_none=True)
        with torch.autocast(target.type,dtype=torch.bfloat16,enabled=target.type=="cuda"): loss,pieces=_loss(model,batch)
        loss.backward(); gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        if not math.isfinite(float(gradient)): raise FloatingPointError("grounded feedback gradient became non-finite")
        optimizer.step()
        with torch.no_grad():
            for e,v in zip(ema.parameters(),model.parameters(),strict=True): e.lerp_(v,1-config.ema_decay)
        if update==1 or update%100==0 or update==config.updates:
            row={"update":update,**{k:round(v,8) for k,v in pieces.items()},"gradient":round(float(gradient),8)}
            history.append(row); print(json.dumps(row),flush=True)
    seconds=time.perf_counter()-started; candidates={}
    for name,candidate in (("raw",model.eval()),("ema",ema.eval())):
        offline=_offline(candidate,validation,target); rollout=_rollout(candidate,validation,target)
        candidates[name]={**offline,**rollout}
    selected_name=max(candidates,key=lambda n:(int(_gates(candidates[n])["all_passed"]),candidates[n]["contact_f1"]-candidates[n]["muscle_mae"]))
    selected=model if selected_name=="raw" else ema; metrics=candidates[selected_name]; gates=_gates(metrics)
    state={n:v.detach().cpu().clone() for n,v in selected.state_dict().items()}
    report={"format":FORMAT,"status":"passed" if gates["all_passed"] else "failed-quality","source_sha256":source_sha256(),
            "train_corpus_sha256":train_corpus.semantic_sha256,"validation_corpus_sha256":validation.semantic_sha256,
            "model_config":model_config.to_dict(),"training_config":config.to_dict(),"parameters":model.parameter_count,
            "selected_weights":selected_name,"candidate_metrics":candidates,"metrics":metrics,"gates":gates,"history":history,
            "runtime":{"device":str(target),"seconds":seconds,"updates_per_second":config.updates/seconds,
                       "peak_allocated_bytes":int(torch.cuda.max_memory_allocated(target)) if target.type=="cuda" else 0}}
    payload={"format":CHECKPOINT_FORMAT,"source_sha256":report["source_sha256"],"model_config":model_config.to_dict(),
             "model_state":state,"model_state_sha256":_state_sha256(state),"report":report}
    stage=output.parent/f".{output.name}.tmp-{uuid.uuid4().hex}"; stage.mkdir(parents=True)
    checkpoint=stage/"runtime.pt"; torch.save(payload,checkpoint); raw=checkpoint.read_bytes()
    report["checkpoint"]={"path":"runtime.pt","bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest(),"model_state_sha256":payload["model_state_sha256"]}
    (stage/"report.json").write_text(json.dumps(report,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    stage.rename(output); return report
