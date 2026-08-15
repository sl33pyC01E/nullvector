from __future__ import annotations
import hashlib,math,os,time,uuid
from pathlib import Path
import numpy as np
import torch
from ..config import PROJECT_ROOT
from ..safety import require_disk_floor
from ..creature_stage_neural_motion.contract import CellularMotionTransformerConfig
from ..creature_stage_neural_motion.training import _state_sha256
from ..creature_stage_neural_grounded.contract import CHECKPOINT_FORMAT,GroundedModelConfig,canonical_json_bytes,sha256_file
from ..creature_stage_neural_grounded.dataset import GroundedMotionTeacher
from ..creature_stage_neural_grounded.model import NeuralGroundedMotion
from ..creature_stage_neural_grounded_refine.training import _batch_at,_forward,_l1_loss

FORMAT="nullvector-neural-grounded-top-tail-seam-v4";PARENT=PROJECT_ROOT/"outputs/creature_stage_neural_grounded_continuation/production_v3/grounded_motion_continued_0001100.pt";PARENT_SHA="fb3e69f87f8a5def3977b57f0df74e84f25ae4bf8425d8dfc5a87938b41c529d";FILES=("forge/creature_stage_neural_grounded_seam/__init__.py","forge/creature_stage_neural_grounded_seam/__main__.py","forge/creature_stage_neural_grounded_seam/training.py")
def source_sha256():
 d=hashlib.sha256(b"nullvector-grounded-seam-v4\0"+PARENT_SHA.encode())
 for r in FILES:p=PROJECT_ROOT/r;d.update(r.encode()+b"\0"+p.read_bytes()+b"\0")
 return d.hexdigest()
def _top_tail_seam(model,teacher,device):
 left,right=_batch_at(teacher,71,device),_batch_at(teacher,0,device)
 with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):ol=_forward(model,left,left["state"]);orr=_forward(model,right,ol.cells.detach())
 error=(((orr.cells[:,:,:2].float()-ol.cells[:,:,:2].float())-(right["target"][:,:,:2]-left["target"][:,:,:2])).abs()*right["mask"][:,:,None]).flatten();active=error[error>0];k=max(1,int(active.numel()*.05));return torch.topk(active,k).values.mean()
def train(output:Path,*,updates:int=250,device:str="cuda"):
 output=Path(output).resolve()
 if output.exists():raise FileExistsError(output)
 if not 100<=updates<=600:raise ValueError("seam update count drifted")
 require_disk_floor(output.parent,floor_gb=100,planned_bytes=2*1024**3)
 if sha256_file(PARENT)!=PARENT_SHA:raise ValueError("seam parent bytes drifted")
 parent=torch.load(PARENT,map_location="cpu",weights_only=True)
 if parent.get("format")!=CHECKPOINT_FORMAT or parent.get("continuation",{}).get("format")!="nullvector-neural-grounded-cell-motion-continuation-v3":raise ValueError("seam parent provenance drifted")
 device=torch.device(device);teacher=GroundedMotionTeacher();model=NeuralGroundedMotion(CellularMotionTransformerConfig(**parent["backbone"]),GroundedModelConfig(**parent["model"]));model.load_state_dict(parent["ema_state"],strict=True);model.to(device).train();optimizer=torch.optim.AdamW([{"params":model.backbone.parameters(),"lr":2e-7},{"params":[p for n,p in model.named_parameters() if not n.startswith("backbone.")],"lr":8e-6}],weight_decay=1e-5);ema={n:v.detach().cpu().clone() for n,v in model.state_dict().items()};history=[];started=time.perf_counter()
 for update in range(1,updates+1):
  frames=teacher.sequence(update+1100,10,6,device,split="all");state=frames[0]["state"];optimizer.zero_grad(set_to_none=True);losses=[];sums={}
  for batch in frames:
   with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):out=_forward(model,batch,state)
   loss,pieces=_l1_loss(out,batch,state);losses.append(loss)
   for n,v in pieces.items():sums[n]=sums.get(n,0)+v
   state=out.cells.detach()
  seam=_top_tail_seam(model,teacher,device);losses.append(seam*.55);loss=torch.stack(losses).mean();loss.backward();gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),1)
  if not math.isfinite(float(gradient)):raise FloatingPointError("seam gradient non-finite")
  optimizer.step()
  with torch.no_grad():
   for n,v in model.state_dict().items():ema[n].mul_(.97).add_(v.detach().cpu(),alpha=.03)
  if update==1 or update%25==0 or update==updates:history.append({"update":update,"loss":round(float(loss),9),**{n:round(v/6,9) for n,v in sums.items()},"top_tail_seam_l1":round(float(seam),9),"gradient_norm":round(float(gradient),9)})
 seconds=time.perf_counter()-started;state={n:v.detach().cpu().clone() for n,v in model.state_dict().items()};checkpoint={**{k:parent[k] for k in ("format","source_sha256","teacher_semantic_sha256","parent","model","backbone","training")},"updates":1100+updates,"model_state":state,"ema_state":ema,"model_state_sha256":_state_sha256(state),"ema_state_sha256":_state_sha256(ema),"history":history,"seam_refine":{"format":FORMAT,"source_sha256":source_sha256(),"parent_sha256":PARENT_SHA,"updates":updates}}
 stage=output.parent/f".{output.name}.tmp-{uuid.uuid4().hex}";stage.mkdir(parents=True);cp=stage/f"grounded_motion_seam_{1100+updates:07d}.pt";torch.save(checkpoint,cp);report={"format":FORMAT,"source_sha256":source_sha256(),"parent_sha256":PARENT_SHA,"updates":updates,"total_refine_updates":1100+updates,"checkpoint":{"path":cp.name,"bytes":cp.stat().st_size,"sha256":sha256_file(cp),"model_state_sha256":checkpoint["model_state_sha256"],"ema_state_sha256":checkpoint["ema_state_sha256"]},"history":history,"runtime":{"seconds":round(seconds,6),"updates_per_second":round(updates/seconds,6),"device":str(device)}};report["semantic_sha256"]=hashlib.sha256(canonical_json_bytes(report)).hexdigest();(stage/"production_manifest.json").write_bytes(canonical_json_bytes(report));os.replace(stage,output);return report
