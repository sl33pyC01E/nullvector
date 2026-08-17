from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from ..recurrent_world_student_v5.model import PerceptionRecurrentWorldStudent
from ..world_action_natural_v10 import load
from ..world_latent_dit.contract import ModelConfig
from .contract import CHECKPOINT_FORMAT,CORPUS,DEFAULT_OUTPUT,PARENT,PARENT_SHA256,REPORT_FORMAT,canonical,file_sha256,source_sha256,state_sha256
from .training import _normalizers,_rollout_metrics


def evaluate(output:Path=DEFAULT_OUTPUT):
    output=Path(output).resolve();milestones=sorted(output.glob("milestone_*.pt"))
    if not milestones:raise FileNotFoundError("no V6 milestones")
    sequences,manifest=load(CORPUS);device=torch.device("cuda:0");rows=[]
    for path in milestones:
        payload=torch.load(path,map_location="cpu",weights_only=True)
        if payload.get("format")!=CHECKPOINT_FORMAT or payload.get("source_sha256")!=source_sha256() or payload.get("corpus_sha256")!=manifest["manifest_sha256"] or payload.get("parent_sha256")!=PARENT_SHA256:raise ValueError("V6 milestone drifted")
        rows.append((payload["selection_score"],payload["update"],path,payload))
    _,update,path,payload=min(rows,key=lambda row:(row[0],row[1]));model=PerceptionRecurrentWorldStudent(ModelConfig(**payload["model_config"]));model.load_state_dict(payload["ema_state"]);model.to(device).eval();norms=_normalizers(payload,device);test={str(h):_rollout_metrics(model,sequences[5],norms,device,h,samples=48) for h in (1,2,4,8,16,32)};ablations={mode:_rollout_metrics(model,sequences[5],norms,device,8,perception=mode,samples=48) for mode in ("zero","shuffle")};gates={"all_horizons_beat_persistence":all(row["improvement"]>0 for row in test.values()),"perception_used_at_horizon_8":test["8"]["mae"]<min(row["mae"] for row in ablations.values()),"under_half_gpu_memory":payload["runtime"]["peak_reserved_bytes"]<12*1024**3};gates["all_passed"]=all(gates.values());state=payload["ema_state"];release={"format":CHECKPOINT_FORMAT,"status":"ready" if gates["all_passed"] else "experimental","source_sha256":source_sha256(),"corpus_sha256":manifest["manifest_sha256"],"parent_sha256":PARENT_SHA256,"selected_milestone":{"update":update,"sha256":file_sha256(path)},"model_config":payload["model_config"],"normalization":payload["normalization"],"state":state,"state_sha256":state_sha256(state),"validation":payload["validation"],"test":test,"ablations":ablations,"gates":gates,"plan":payload["plan"],"runtime":payload["runtime"]};temporary=output/".runtime.pt.tmp";torch.save(release,temporary);temporary.replace(output/"runtime.pt");report={k:v for k,v in release.items() if k not in ("state","normalization")};report["format"]=REPORT_FORMAT;report["checkpoint_sha256"]=file_sha256(output/"runtime.pt");report["report_sha256"]=hashlib.sha256(canonical(report)).hexdigest();(output/"evaluation.json").write_bytes(canonical(report));return report
