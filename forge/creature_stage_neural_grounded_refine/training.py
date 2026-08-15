from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import time
import uuid
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..config import PROJECT_ROOT
from ..safety import require_disk_floor
from ..creature_stage_neural_motion.contract import CellularMotionTransformerConfig
from ..creature_stage_neural_motion.training import _state_sha256
from ..creature_stage_neural_grounded.contract import CHECKPOINT_FORMAT as V1_CHECKPOINT_FORMAT, GroundedModelConfig, canonical_json_bytes, sha256_file
from ..creature_stage_neural_grounded.dataset import GroundedMotionTeacher
from ..creature_stage_neural_grounded.model import NeuralGroundedMotion
from .contract import DEFAULT_OUTPUT, FORMAT, PARENT, PARENT_SHA256, PARENT_SOURCE_SHA256, RefineConfig, source_sha256


def _batch_at(teacher: GroundedMotionTeacher, frame: int, device: torch.device) -> dict[str, torch.Tensor]:
    rows = [teacher.sample(identity, frame) for identity in range(10)]
    result = {name: torch.from_numpy(np.stack([row[name] for row in rows]).copy()).to(device) for name in ("static","state","target","dynamic","mask","adjacency","controls")}
    for name in ("family","morphotype","motion"):
        result[name] = torch.tensor([int(row[name]) for row in rows], dtype=torch.long, device=device)
    result["phase"] = torch.tensor([float(row["phase"]) for row in rows], device=device)
    result["body_target"] = torch.tensor([float(row["body_target"]) for row in rows], device=device)
    return result


def _mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    active = mask[:, :, None].to(value.dtype)
    return (value * active).sum() / (active.sum().clamp_min(1) * value.shape[-1])


def _forward(model: NeuralGroundedMotion, batch: dict[str, torch.Tensor], state: torch.Tensor):
    return model(batch["static"], state, batch["dynamic"], batch["mask"], batch["adjacency"], batch["family"], batch["morphotype"], batch["motion"], batch["phase"], batch["controls"])


def _l1_loss(output, batch: dict[str, torch.Tensor], state: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    mask=batch["mask"]; target=batch["target"].float(); predicted=output.cells.float()
    pos=_mean((predicted[:,:,:2]-target[:,:,:2]).abs(),mask); vel=_mean((predicted[:,:,2:]-target[:,:,2:]).abs(),mask)
    app=(batch["static"][:,:,50]>.5)&mask; contact=(batch["dynamic"][:,:,5]>.5)&mask
    app_l1=_mean((predicted[:,:,:2]-target[:,:,:2]).abs(),app); contact_l1=_mean((predicted[:,:,:2]-target[:,:,:2]).abs(),contact) if bool(contact.any()) else pos*0
    direct=_mean((output.direct_cells.float()-target).abs(),mask)
    delta=_mean((output.delta_cells[:,:,:2].float()-(target[:,:,:2]-state[:,:,:2].float())).abs(),mask)
    body=(output.body_velocity.float()-batch["body_target"].float()).abs().mean()
    total=pos*1.20+vel*.28+app_l1*.55+contact_l1*.85+direct*.80+delta*.18+body*.25
    return total,{"position_l1":float(pos),"velocity_l1":float(vel),"appendage_l1":float(app_l1),"contact_l1":float(contact_l1),"direct_l1":float(direct),"delta_l1":float(delta),"body_l1":float(body)}


def train(output: Path = DEFAULT_OUTPUT, *, updates: int | None = None, device: str = "cuda") -> dict[str, Any]:
    output=Path(output).resolve()
    if output.exists(): raise FileExistsError(output)
    base=RefineConfig(); config=RefineConfig(updates=base.updates if updates is None else updates)
    require_disk_floor(output.parent,floor_gb=100,planned_bytes=2*1024**3)
    target_device=torch.device(device); teacher=GroundedMotionTeacher()
    if sha256_file(PARENT)!=PARENT_SHA256: raise ValueError("grounded refine parent bytes drifted")
    parent=torch.load(PARENT,map_location="cpu",weights_only=True)
    if parent.get("format")!=V1_CHECKPOINT_FORMAT or parent.get("source_sha256")!=PARENT_SOURCE_SHA256: raise ValueError("grounded refine parent contract drifted")
    model=NeuralGroundedMotion(CellularMotionTransformerConfig(**parent["backbone"]),GroundedModelConfig(**parent["model"]))
    model.load_state_dict(parent["ema_state"],strict=True); model.to(target_device).train()
    optimizer=torch.optim.AdamW([{"params":model.backbone.parameters(),"lr":config.learning_rate*config.backbone_scale},{"params":[p for n,p in model.named_parameters() if not n.startswith("backbone.")],"lr":config.learning_rate}],weight_decay=1e-5)
    ema={n:v.detach().cpu().clone() for n,v in model.state_dict().items()}; history=[]; started=time.perf_counter()
    for update in range(1,config.updates+1):
        frames=teacher.sequence(update,config.batch_size,config.sequence_frames,target_device,split="all"); state=frames[0]["state"]; optimizer.zero_grad(set_to_none=True); losses=[]; sums={}
        for batch in frames:
            with torch.autocast(target_device.type,dtype=torch.bfloat16,enabled=target_device.type=="cuda"): out=_forward(model,batch,state)
            loss,pieces=_l1_loss(out,batch,state); losses.append(loss)
            for n,v in pieces.items(): sums[n]=sums.get(n,0.0)+v
            state=out.cells.detach()
        seam_value=0.0
        if update%config.seam_every==0:
            left,right=_batch_at(teacher,71,target_device),_batch_at(teacher,0,target_device)
            with torch.autocast(target_device.type,dtype=torch.bfloat16,enabled=target_device.type=="cuda"):
                out_left=_forward(model,left,left["state"]); out_right=_forward(model,right,out_left.cells.detach())
            pred_transition=out_right.cells[:,:,:2].float()-out_left.cells[:,:,:2].float(); target_transition=right["target"][:,:,:2]-left["target"][:,:,:2]
            seam=_mean((pred_transition-target_transition).abs(),right["mask"]); losses.append(seam*.25); seam_value=float(seam)
        loss=torch.stack(losses).mean();loss.backward();gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        if not math.isfinite(float(gradient)): raise FloatingPointError("grounded refine gradient non-finite")
        optimizer.step()
        with torch.no_grad():
            for n,v in model.state_dict().items(): ema[n].mul_(config.ema_decay).add_(v.detach().cpu(),alpha=1-config.ema_decay)
        if update==1 or update%25==0 or update==config.updates: history.append({"update":update,"loss":round(float(loss),9),**{n:round(v/len(frames),9) for n,v in sums.items()},"seam_l1":round(seam_value,9),"gradient_norm":round(float(gradient),9)})
    seconds=time.perf_counter()-started; model_state={n:v.detach().cpu().clone() for n,v in model.state_dict().items()}
    # Remain loader-compatible with the unchanged v1 architecture while the
    # separate production manifest records the v2 training provenance.
    checkpoint={**{k:parent[k] for k in ("format","source_sha256","teacher_semantic_sha256","parent","model","backbone","training")},"updates":config.updates,"model_state":model_state,"ema_state":ema,"model_state_sha256":_state_sha256(model_state),"ema_state_sha256":_state_sha256(ema),"history":history,"refine":{"format":FORMAT,"source_sha256":source_sha256(),"parent_sha256":PARENT_SHA256,"config":config.to_dict()}}
    stage=output.parent/f".{output.name}.tmp-{uuid.uuid4().hex}";stage.mkdir(parents=True);cp=stage/f"grounded_motion_refined_{config.updates:07d}.pt";torch.save(checkpoint,cp)
    report={"format":FORMAT,"source_sha256":source_sha256(),"parent":{"path":PARENT.relative_to(PROJECT_ROOT).as_posix(),"sha256":PARENT_SHA256,"ema_state_sha256":parent["ema_state_sha256"]},"teacher_semantic_sha256":teacher.semantic_sha256,"config":config.to_dict(),"checkpoint":{"path":cp.name,"bytes":cp.stat().st_size,"sha256":sha256_file(cp),"model_state_sha256":checkpoint["model_state_sha256"],"ema_state_sha256":checkpoint["ema_state_sha256"]},"history":history,"runtime":{"seconds":round(seconds,6),"updates_per_second":round(config.updates/seconds,6),"device":str(target_device),"peak_allocated_bytes":torch.cuda.max_memory_allocated() if target_device.type=="cuda" else 0}}
    report["semantic_sha256"]=hashlib.sha256(canonical_json_bytes(report)).hexdigest();(stage/"production_manifest.json").write_bytes(canonical_json_bytes(report));os.replace(stage,output);return report
