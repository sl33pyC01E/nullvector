from __future__ import annotations
import hashlib, math, os, time, uuid
from pathlib import Path
import numpy as np
import torch
from ..config import PROJECT_ROOT
from ..safety import require_disk_floor
from ..creature_stage_neural_motion.contract import CellularMotionTransformerConfig
from ..creature_stage_neural_motion.training import _state_sha256
from ..creature_stage_neural_grounded.contract import CHECKPOINT_FORMAT, GroundedModelConfig, canonical_json_bytes, sha256_file
from ..creature_stage_neural_grounded.dataset import GroundedMotionTeacher
from ..creature_stage_neural_grounded.model import NeuralGroundedMotion
from ..creature_stage_neural_grounded_refine.training import _batch_at, _forward, _l1_loss

FORMAT="nullvector-neural-grounded-cell-motion-continuation-v3"
PARENT=PROJECT_ROOT/"outputs/creature_stage_neural_grounded_refine/production_v2_800/grounded_motion_refined_0000800.pt"
PARENT_SHA="c93d71cb2de15f0c5d7130ea9ee41b979a4e9d99628670243ec5a89112a076ef"
SOURCE_FILES=("forge/creature_stage_neural_grounded_continuation/__init__.py","forge/creature_stage_neural_grounded_continuation/__main__.py","forge/creature_stage_neural_grounded_continuation/training.py")

def source_sha256():
    d=hashlib.sha256(b"nullvector-grounded-continuation-v3\0"+PARENT_SHA.encode())
    for relative in SOURCE_FILES:
        path=PROJECT_ROOT/relative;d.update(relative.encode()+b"\0"+path.read_bytes()+b"\0")
    return d.hexdigest()

def train(output:Path,*,updates:int=300,device:str="cuda"):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    if not 100<=updates<=1000:raise ValueError("continuation update count drifted")
    require_disk_floor(output.parent,floor_gb=100,planned_bytes=2*1024**3)
    if sha256_file(PARENT)!=PARENT_SHA:raise ValueError("continuation parent bytes drifted")
    parent=torch.load(PARENT,map_location="cpu",weights_only=True)
    if parent.get("format")!=CHECKPOINT_FORMAT or parent.get("refine",{}).get("source_sha256")!="025aeca295326dece9de13978e1eb0b02ffe86ee01d4ee6facdd30ae7e64ef53":raise ValueError("continuation parent provenance drifted")
    target_device=torch.device(device);teacher=GroundedMotionTeacher();model=NeuralGroundedMotion(CellularMotionTransformerConfig(**parent["backbone"]),GroundedModelConfig(**parent["model"]));model.load_state_dict(parent["ema_state"],strict=True);model.to(target_device).train()
    optimizer=torch.optim.AdamW([{"params":model.backbone.parameters(),"lr":5e-7},{"params":[p for n,p in model.named_parameters() if not n.startswith("backbone.")],"lr":1e-5}],weight_decay=1e-5);ema={n:v.detach().cpu().clone() for n,v in model.state_dict().items()};history=[];started=time.perf_counter()
    for update in range(1,updates+1):
        frames=teacher.sequence(update+800,10,6,target_device,split="all");state=frames[0]["state"];optimizer.zero_grad(set_to_none=True);losses=[];sums={}
        for batch in frames:
            with torch.autocast(target_device.type,dtype=torch.bfloat16,enabled=target_device.type=="cuda"):out=_forward(model,batch,state)
            loss,pieces=_l1_loss(out,batch,state);losses.append(loss)
            for n,v in pieces.items():sums[n]=sums.get(n,0)+v
            state=out.cells.detach()
        seam_value=0.0
        if update%2==0:
            left,right=_batch_at(teacher,71,target_device),_batch_at(teacher,0,target_device)
            with torch.autocast(target_device.type,dtype=torch.bfloat16,enabled=target_device.type=="cuda"):ol=_forward(model,left,left["state"]);orr=_forward(model,right,ol.cells.detach())
            seam=((orr.cells[:,:,:2].float()-ol.cells[:,:,:2].float())-(right["target"][:,:,:2]-left["target"][:,:,:2])).abs();active=right["mask"][:,:,None];seam=(seam*active).sum()/(active.sum()*2);losses.append(seam*.2);seam_value=float(seam)
        loss=torch.stack(losses).mean();loss.backward();gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),1)
        if not math.isfinite(float(gradient)):raise FloatingPointError("continuation gradient non-finite")
        optimizer.step()
        with torch.no_grad():
            for n,v in model.state_dict().items():ema[n].mul_(.98).add_(v.detach().cpu(),alpha=.02)
        if update==1 or update%25==0 or update==updates:history.append({"update":update,"loss":round(float(loss),9),**{n:round(v/6,9) for n,v in sums.items()},"seam_l1":round(seam_value,9),"gradient_norm":round(float(gradient),9)})
    seconds=time.perf_counter()-started;state={n:v.detach().cpu().clone() for n,v in model.state_dict().items()};checkpoint={**{k:parent[k] for k in ("format","source_sha256","teacher_semantic_sha256","parent","model","backbone","training")},"updates":800+updates,"model_state":state,"ema_state":ema,"model_state_sha256":_state_sha256(state),"ema_state_sha256":_state_sha256(ema),"history":history,"continuation":{"format":FORMAT,"source_sha256":source_sha256(),"parent_sha256":PARENT_SHA,"updates":updates}}
    stage=output.parent/f".{output.name}.tmp-{uuid.uuid4().hex}";stage.mkdir(parents=True);cp=stage/f"grounded_motion_continued_{800+updates:07d}.pt";torch.save(checkpoint,cp);report={"format":FORMAT,"source_sha256":source_sha256(),"parent_sha256":PARENT_SHA,"updates":updates,"total_refine_updates":800+updates,"checkpoint":{"path":cp.name,"bytes":cp.stat().st_size,"sha256":sha256_file(cp),"model_state_sha256":checkpoint["model_state_sha256"],"ema_state_sha256":checkpoint["ema_state_sha256"]},"history":history,"runtime":{"seconds":round(seconds,6),"updates_per_second":round(updates/seconds,6),"device":str(target_device)}};report["semantic_sha256"]=hashlib.sha256(canonical_json_bytes(report)).hexdigest();(stage/"production_manifest.json").write_bytes(canonical_json_bytes(report));os.replace(stage,output);return report
