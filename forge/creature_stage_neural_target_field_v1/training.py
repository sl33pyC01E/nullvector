from __future__ import annotations
import copy,hashlib,json,math,time,uuid
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from ..creature_stage_neural_motion.training import _state_sha256
from ..safety import require_disk_floor
from .contract import *
from .dataset import build_target_augmentation,build_target_corpus
from .model import NeuralGroundedTargetField
from .runtime import NeuralTargetFieldRuntime
from .physics import simulate_target_field_cycle
from .bank import load_training_bank

def _target_loss(output,target,mask):
 per_owner=F.smooth_l1_loss(output,target.float(),beta=.015,reduction="none").mean(-1)[mask]
 count=max(1,int(per_owner.numel())//4)
 return per_owner.mean()+per_owner.topk(count).values.mean()*.75

def loss_fn(model,b,target_aug=None):
 o=model(b["owner_state"],b["global_state"],b["owner_mask"],b["muscle_meta"],b["muscle_owner"],b["muscle_mask"],b["target_context"]);mask=b["owner_mask"][:,:,None]
 target=_target_loss(o.terminal_target,b["terminal_target"],b["owner_mask"])
 augmented=torch.zeros((),device=target.device)
 if target_aug is not None:
  a=model(target_aug["owner_state"],target_aug["global_state"],target_aug["owner_mask"],target_aug["muscle_meta"],target_aug["muscle_owner"],target_aug["muscle_mask"],target_aug["target_context"])
  augmented=_target_loss(a.terminal_target,target_aug["terminal_target"],target_aug["owner_mask"])
 muscle=F.smooth_l1_loss(o.feedback.muscle_activation[b["muscle_mask"]],b["muscle_target"].float()[b["muscle_mask"]],beta=.04)
 logits=o.feedback.contact_logits[b["owner_mask"]];truth=b["contact_target"].float()[b["owner_mask"]];contact=F.binary_cross_entropy_with_logits(logits,truth)
 total=(target+augmented)*6+muscle*1.5+contact*.5
 return total,{"loss":float(total.detach()),"target":float(target.detach()),"target_aug":float(augmented.detach()),"muscle":float(muscle.detach()),"contact":float(contact.detach())}

@torch.inference_mode()
def evaluate(model,corpus,device):
 errors=[];tp=fp=fn=0
 for start in range(0,corpus.samples,1024):
  ids=torch.arange(start,min(start+1024,corpus.samples));b=corpus.batch(ids,device);o=model(b["owner_state"],b["global_state"],b["owner_mask"],b["muscle_meta"],b["muscle_owner"],b["muscle_mask"],b["target_context"])
  errors.append(((o.terminal_target-b["terminal_target"].float()).norm(dim=-1)*24)[b["owner_mask"]].cpu());p=(torch.sigmoid(o.feedback.contact_logits)>=.5)&b["owner_mask"];t=(b["contact_target"]>.5)&b["owner_mask"];tp+=int((p&t).sum());fp+=int((p&~t).sum());fn+=int((~p&t).sum())
 runtime=NeuralTargetFieldRuntime(model,device);ratios=[];slip=strain=vertical=seam=node=0.
 for fam in range(5):
  i=next(i for i,x in enumerate(corpus.feedback.organisms) if int(np.argmax(x.genome.family_mix))==fam);teacher=corpus.feedback.cycles[i];cycle=simulate_target_field_cycle(corpus.feedback.organisms[i],runtime);ratios.append(cycle.distance_px/teacher.distance_px);slip=max(slip,cycle.maximum_contact_slip_px);strain=max(strain,cycle.maximum_edge_strain);vertical=max(vertical,cycle.vertical_axis_max_degrees);seam=max(seam,cycle.loop_seam_max_abs);node+=np.mean([np.abs(a.nodes_local-b.nodes_local).mean() for a,b in zip(cycle.frames,teacher.frames)])
 e=torch.cat(errors);return {"terminal_mae_px":float(e.mean()),"terminal_p95_px":float(torch.quantile(e,.95)),"contact_f1":2*tp/max(2*tp+fp+fn,1),"advance_ratio_min":float(min(ratios)),"advance_ratio_max":float(max(ratios)),"node_l1":float(node/5),"maximum_contact_slip_px":slip,"maximum_edge_strain":strain,"vertical_axis_max_degrees":vertical,"loop_seam_max_abs":seam}

def gates(m):
 g={"target_accuracy":m["terminal_mae_px"]<.18 and m["terminal_p95_px"]<.45,"contact_accuracy":m["contact_f1"]>.98,"advance":m["advance_ratio_min"]>.75 and m["advance_ratio_max"]<1.3,"shape":m["node_l1"]<.7,"physics":m["maximum_contact_slip_px"]<.05 and m["maximum_edge_strain"]<.12 and m["vertical_axis_max_degrees"]<5 and m["loop_seam_max_abs"]<.002};g["all_passed"]=all(g.values());return g

def train(output:Path,*,updates=None,device="cuda",bank:Path|None=None):
 output=Path(output).resolve();require_disk_floor(output.parent,floor_gb=100,planned_bytes=2*1024**3)
 if output.exists():raise FileExistsError(output)
 cfg=TrainingConfig(updates=updates or TrainingConfig().updates);dev=torch.device(device if device!="cuda" or torch.cuda.is_available() else "cpu")
 if bank is None: trainset=build_target_corpus(split="train",variants_per_family=cfg.variants_per_family);target_aug=build_target_augmentation(variants_per_chassis=cfg.target_variants_per_chassis)
 else: trainset,target_aug=load_training_bank(bank)
 val=build_target_corpus(split="validation")
 torch.manual_seed(cfg.seed);np.random.seed(cfg.seed&0xffffffff);model_cfg=ModelConfig();model=NeuralGroundedTargetField(model_cfg).to(dev).train();ema=copy.deepcopy(model).eval();opt=torch.optim.AdamW(model.parameters(),lr=cfg.learning_rate,fused=dev.type=="cuda");gen=torch.Generator().manual_seed(cfg.seed);history=[];started=time.perf_counter()
 for u in range(1,cfg.updates+1):
  b=trainset.batch(torch.randint(0,trainset.samples,(cfg.batch_size,),generator=gen),dev);a=target_aug.batch(torch.randint(0,target_aug.samples,(cfg.batch_size,),generator=gen),dev);opt.zero_grad(set_to_none=True)
  with torch.autocast(dev.type,dtype=torch.bfloat16,enabled=dev.type=="cuda"):loss,pieces=loss_fn(model,b,a)
  loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
  with torch.no_grad():
   for a,v in zip(ema.parameters(),model.parameters(),strict=True):a.lerp_(v,1-cfg.ema_decay)
  if u==1 or u%100==0 or u==cfg.updates:row={"update":u,**{k:round(v,8) for k,v in pieces.items()},"gradient":round(float(grad),8)};history.append(row);print(json.dumps(row),flush=True)
 candidates={n:evaluate(m.eval(),val,dev) for n,m in (("raw",model),("ema",ema))};selected=max(candidates,key=lambda n:(int(gates(candidates[n])["all_passed"]),-candidates[n]["terminal_mae_px"]));chosen=model if selected=="raw" else ema;metrics=candidates[selected];gate=gates(metrics);state={n:v.detach().cpu().clone() for n,v in chosen.state_dict().items()};report={"format":FORMAT,"status":"passed" if gate["all_passed"] else "failed-quality","source_sha256":source_sha256(),"model_config":model_cfg.to_dict(),"training_config":cfg.to_dict(),"corpora":{"feedback":trainset.semantic_sha256,"target_augmentation":target_aug.semantic_sha256,"validation":val.semantic_sha256},"parameters":model.parameter_count,"selected_weights":selected,"metrics":metrics,"candidate_metrics":candidates,"gates":gate,"history":history,"runtime":{"seconds":time.perf_counter()-started,"device":str(dev)}};payload={"format":CHECKPOINT_FORMAT,"source_sha256":report["source_sha256"],"model_config":model_cfg.to_dict(),"model_state":state,"model_state_sha256":_state_sha256(state),"report":report};stage=output.parent/f".{output.name}.tmp-{uuid.uuid4().hex}";stage.mkdir(parents=True);p=stage/"runtime.pt";torch.save(payload,p);raw=p.read_bytes();report["checkpoint"]={"sha256":hashlib.sha256(raw).hexdigest(),"bytes":len(raw)};(stage/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n");stage.rename(output);return report
