from __future__ import annotations
import copy,hashlib,json,os,time
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from ..safety import require_disk_floor
from ..world_action_cellular_v7.corpus import load_encoded_corpus
from ..world_latent_dit.contract import ModelConfig
from .contract import BASE_CHECKPOINT,BASE_SHA256,CHECKPOINT_FORMAT,CORPUS,DEFAULT_OUTPUT,REPORT_FORMAT,TrainingPlan,canonical,file_sha256,source_sha256,state_sha256
from .model import RecurrentActionDiT

def _atomic_torch(path,payload):temporary=path.with_name(f".{path.name}.tmp-{os.getpid()}");torch.save(payload,temporary);os.replace(temporary,path)
def _tensor(value,device,dtype=torch.float32):return torch.as_tensor(value,dtype=dtype,device=device)
def _sample(episodes,rng,batch):
    rows=[]
    for _ in range(batch):
        episode=episodes[int(rng.integers(0,len(episodes)))];index=int(rng.integers(0,len(episode["current"])));rows.append((episode,index))
    names=("previous","current","target","action","control","state","actor_state")
    return {name:np.stack([episode[name][index] for episode,index in rows]) for name in names}
@torch.inference_mode()
def _raw_delta(model,episode,device,mean,std,wrong=False,batch=8):
    rows=[]
    for start in range(0,len(episode["current"]),batch):
        end=start+batch;current=_tensor(episode["current"][start:end],device);previous=_tensor(episode["previous"][start:end],device);cn=(current-mean)/std;pn=(previous-mean)/std;action=_tensor(episode["action"][start:end],device,torch.long);action=(action+7)%22 if wrong else action;rows.append(model(cn,pn,action,_tensor(episode["control"][start:end],device),_tensor(episode["state"][start:end],device),_tensor(episode["actor_state"][start:end],device)).float().cpu())
    return torch.cat(rows)
def _metrics_from_delta(episode,delta,wrong_delta,std,threshold,alpha):
    current=torch.from_numpy(episode["current"]);target=torch.from_numpy(episode["target"]);scale=std.detach().cpu();gate=delta.abs().mean(1,keepdim=True)>=threshold;wrong_gate=wrong_delta.abs().mean(1,keepdim=True)>=threshold;prediction=current+alpha*gate*delta*scale;wrong=current+alpha*wrong_gate*wrong_delta*scale;mae=float(F.l1_loss(prediction,target));persistence=float(F.l1_loss(current,target));return {"mae":mae,"persistence_mae":persistence,"improvement":1-mae/persistence,"action_advantage":float(F.l1_loss(wrong,target))-mae}
def _metrics(model,episode,device,mean,std,threshold,alpha):return _metrics_from_delta(episode,_raw_delta(model,episode,device,mean,std),_raw_delta(model,episode,device,mean,std,wrong=True),std,threshold,alpha)
def _calibrate(model,episode,device,mean,std):
    best=None;delta=_raw_delta(model,episode,device,mean,std);wrong=_raw_delta(model,episode,device,mean,std,wrong=True)
    for threshold in (0.,.01,.02,.03,.05,.08,.12,.18):
        for alpha in (.1,.2,.35,.5,.7,.85,1.):
            row=_metrics_from_delta(episode,delta,wrong,std,threshold,alpha);score=row["improvement"]+min(.05,row["action_advantage"])*2
            candidate=(score,row["improvement"],row["action_advantage"],-threshold,-alpha,threshold,alpha,row)
            if best is None or candidate[:5]>best[:5]:best=candidate
    return {"threshold":best[5],"alpha":best[6],"metrics":best[7]}
def train(output:Path=DEFAULT_OUTPUT,*,corpus:Path=CORPUS,plan:TrainingPlan=TrainingPlan()):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True);require_disk_floor(output.parent,floor_gb=100,planned_bytes=2*1024**3)
    if file_sha256(BASE_CHECKPOINT)!=BASE_SHA256:raise ValueError("promoted Action-DiT parent drifted")
    if not torch.cuda.is_available():raise RuntimeError("recurrent Action-DiT production requires CUDA")
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.60,0);torch.manual_seed(plan.seed);np.random.seed(plan.seed&0xffffffff);device=torch.device("cuda:0");episodes,manifest=load_encoded_corpus(corpus);train_episodes,validation,test=episodes[:4],episodes[4],episodes[5];base=torch.load(BASE_CHECKPOINT,map_location="cpu",weights_only=True);mean=_tensor(base["latent_mean"],device)[None,:,None,None];std=_tensor(base["latent_std"],device)[None,:,None,None];model=RecurrentActionDiT(ModelConfig(**base["model_config"]));model.backbone.load_state_dict(base["ema_state"],strict=True);model.to(device);ema=copy.deepcopy(model).eval().requires_grad_(False);optimizer=torch.optim.AdamW(model.parameters(),lr=plan.learning_rate,weight_decay=1e-3,fused=True);rng=np.random.default_rng(plan.seed);history=[];start=0
    latest=output/"latest.pt"
    if latest.is_file():
        payload=torch.load(latest,map_location="cpu",weights_only=True)
        if payload.get("format")!=CHECKPOINT_FORMAT or payload.get("source_sha256")!=source_sha256() or payload.get("corpus_sha256")!=manifest["manifest_sha256"]:raise ValueError("recurrent Action-DiT resume drifted")
        model.load_state_dict(payload["model_state"]);ema.load_state_dict(payload["ema_state"]);optimizer.load_state_dict(payload["optimizer_state"]);rng.bit_generator.state=payload["rng_state"];history=list(payload["history"]);start=int(payload["update"])
    for end in range(start+plan.segment_updates,plan.total_updates+1,plan.segment_updates):
        began=time.perf_counter();torch.cuda.reset_peak_memory_stats(device);model.train()
        for update in range(end-plan.segment_updates+1,end+1):
            batch=_sample(train_episodes,rng,plan.batch_size);current=(_tensor(batch["current"],device)-mean)/std;previous=(_tensor(batch["previous"],device)-mean)/std;target=(_tensor(batch["target"],device)-mean)/std;action=_tensor(batch["action"],device,torch.long);control=_tensor(batch["control"],device);state=_tensor(batch["state"],device);actor=_tensor(batch["actor_state"],device);desired=target-current;optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                predicted=model(current,previous,action,control,state,actor);magnitude=desired.detach().abs().mean(1,keepdim=True);weight=1+plan.changed_weight*torch.clamp(magnitude/.35,0,2);base_loss=(F.smooth_l1_loss(predicted,desired,reduction="none")*weight).mean();static=(predicted.abs()*torch.exp(-magnitude*8)).mean();wrong=model(current[:2],previous[:2],(action[:2]+7)%22,control[:2],state[:2],actor[:2]);correct_error=(predicted[:2]-desired[:2]).abs().mean((1,2,3));wrong_error=(wrong-desired[:2]).abs().mean((1,2,3));contrastive=torch.relu(.01+correct_error-wrong_error).mean();loss=base_loss+plan.static_weight*static+plan.contrastive_weight*contrastive
            if not bool(torch.isfinite(loss)):raise FloatingPointError("recurrent Action-DiT loss became non-finite")
            loss.backward();gradient=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1));optimizer.step()
            with torch.no_grad():
                torch._foreach_mul_(list(ema.parameters()),plan.ema_decay);torch._foreach_add_(list(ema.parameters()),list(model.parameters()),alpha=1-plan.ema_decay)
            if update==1 or update%25==0:history.append({"update":update,"loss":round(float(loss),7),"residual":round(float(base_loss),7),"static":round(float(static),7),"contrastive":round(float(contrastive),7),"gradient":round(gradient,7)})
        model.eval();ema.eval();raw_cal=_calibrate(model,validation,device,mean,std);ema_cal=_calibrate(ema,validation,device,mean,std);variant="raw" if raw_cal["metrics"]["improvement"]>=ema_cal["metrics"]["improvement"] else "ema";selected=raw_cal if variant=="raw" else ema_cal;model.train();model_state={n:v.detach().cpu() for n,v in model.state_dict().items()};ema_state={n:v.detach().cpu() for n,v in ema.state_dict().items()};payload={"format":CHECKPOINT_FORMAT,"source_sha256":source_sha256(),"corpus_sha256":manifest["manifest_sha256"],"base_sha256":BASE_SHA256,"model_config":base["model_config"],"plan":plan.to_dict(),"update":end,"model_state":model_state,"ema_state":ema_state,"optimizer_state":optimizer.state_dict(),"rng_state":rng.bit_generator.state,"history":history,"normalization":{"mean":base["latent_mean"],"std":base["latent_std"]},"validation":{"raw":raw_cal,"ema":ema_cal,"selected":variant},"runtime":{"segment_seconds":round(time.perf_counter()-began,6),"peak_reserved_bytes":int(torch.cuda.max_memory_reserved(device))}};_atomic_torch(latest,payload);_atomic_torch(output/f"milestone_{end:07d}.pt",payload);print(json.dumps({"update":end,"selected":variant,"improvement":selected["metrics"]["improvement"],**payload["runtime"]}),flush=True)
    selected_variant=payload["validation"]["selected"];selection=payload["validation"][selected_variant];chosen=model if selected_variant=="raw" else ema;chosen.eval();test_metrics=_metrics(chosen,test,device,mean,std,selection["threshold"],selection["alpha"]);state={n:v.detach().cpu() for n,v in chosen.state_dict().items()};release={"format":CHECKPOINT_FORMAT,"source_sha256":source_sha256(),"corpus_sha256":manifest["manifest_sha256"],"base_sha256":BASE_SHA256,"model_config":base["model_config"],"normalization":{"mean":base["latent_mean"],"std":base["latent_std"]},"state":state,"state_sha256":state_sha256(state)};_atomic_torch(output/"runtime.pt",release);report={"format":REPORT_FORMAT,"status":"ready" if test_metrics["improvement"]>0 and test_metrics["action_advantage"]>0 else "experimental","source_sha256":source_sha256(),"corpus_sha256":manifest["manifest_sha256"],"parameters":chosen.parameter_count,"updates":plan.total_updates,"selection":{"variant":selected_variant,"threshold":selection["threshold"],"alpha":selection["alpha"],"validation":selection["metrics"]},"test":test_metrics,"checkpoint":{"path":"runtime.pt","sha256":file_sha256(output/"runtime.pt"),"state_sha256":release["state_sha256"]},"runtime":payload["runtime"],"gates":{"beats_persistence":test_metrics["improvement"]>0,"correct_action_advantage":test_metrics["action_advantage"]>0}};(output/"report.json").write_bytes(canonical(report));return report
