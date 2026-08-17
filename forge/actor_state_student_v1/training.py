from __future__ import annotations
import copy,hashlib,json,os,time
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from ..safety import require_disk_floor
from ..world_action_cellular_v7.corpus import load_encoded_corpus
from .contract import CHECKPOINT_FORMAT,DEFAULT_CORPUS,DEFAULT_OUTPUT,Plan,REPORT_FORMAT,canonical,source_sha256,state_sha256
from .model import ActorStateStudent
def _hash(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1<<20),b""):digest.update(chunk)
    return digest.hexdigest()
def _group(episodes,name):return np.concatenate([episode[name] for episode in episodes])
def _normalizers(episodes):
    values=np.concatenate((_group(episodes,"actor_state"),_group(episodes,"target_actor_state")));return values.mean(0).astype(np.float32),(values.std(0)+1e-4).astype(np.float32)
def train(output:Path=DEFAULT_OUTPUT,*,corpus:Path=DEFAULT_CORPUS,plan:Plan=Plan()):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True);require_disk_floor(output.parent,floor_gb=100,planned_bytes=256*1024**2)
    if not torch.cuda.is_available():raise RuntimeError("actor student requires CUDA")
    torch.set_num_threads(2);torch.manual_seed(plan.seed);device=torch.device("cuda:0");torch.cuda.set_per_process_memory_fraction(.35,0);episodes,manifest=load_encoded_corpus(corpus);train_episodes=episodes[:4];mean,std=_normalizers(train_episodes);mn=torch.from_numpy(mean).to(device);sd=torch.from_numpy(std).to(device);arrays={name:_group(train_episodes,name) for name in ("previous","current","target","action","control","state","actor_state","target_actor_state")};previous_actor=np.concatenate([np.concatenate((episode["actor_state"][:1],episode["actor_state"][:-1])) for episode in train_episodes]);model=ActorStateStudent().to(device);ema=copy.deepcopy(model).eval().requires_grad_(False);optimizer=torch.optim.AdamW(model.parameters(),lr=plan.learning_rate,weight_decay=1e-4,fused=True);rng=np.random.default_rng(plan.seed);history=[];start=0
    for step in range(plan.segment,plan.updates+1,plan.segment):
        path=output/f"actor_{step:07d}.pt"
        if path.is_file():payload=torch.load(path,map_location="cpu",weights_only=True);start=step
    if start:model.load_state_dict(payload["model_state"]);ema.load_state_dict(payload["ema_state"]);optimizer.load_state_dict(payload["optimizer_state"]);rng.bit_generator.state=payload["rng_state"];history=list(payload["history"])
    for end in range(start+plan.segment,plan.updates+1,plan.segment):
        began=time.perf_counter();torch.cuda.reset_peak_memory_stats(device);model.train()
        for update in range(end-plan.segment+1,end+1):
            indices=rng.integers(0,len(arrays["actor_state"]),plan.batch_size);current=(torch.from_numpy(arrays["actor_state"][indices]).to(device)-mn)/sd;previous=(torch.from_numpy(previous_actor[indices]).to(device)-mn)/sd;target=(torch.from_numpy(arrays["target_actor_state"][indices]).to(device)-mn)/sd;action=torch.from_numpy(arrays["action"][indices].astype(np.int64)).to(device);control=torch.from_numpy(arrays["control"][indices]).to(device);state=torch.from_numpy(arrays["state"][indices]).to(device);changed=((target-current).abs()>.025).float();optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.bfloat16):result=model(current,previous,action,control,state);error=F.smooth_l1_loss(result.state,target,reduction="none");regression=(error*(1+8*changed)).mean();unchanged=((result.state-current).abs()*(1-changed)).mean()
            with torch.autocast("cuda",enabled=False):gate=F.binary_cross_entropy(result.gate.float().clamp(1e-5,1-1e-5),changed)
            loss=regression+.3*gate+.5*unchanged
            loss.backward();gradient=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1));optimizer.step()
            with torch.no_grad():torch._foreach_mul_(list(ema.parameters()),plan.ema_decay);torch._foreach_add_(list(ema.parameters()),list(model.parameters()),alpha=1-plan.ema_decay)
            if update==1 or update%50==0:history.append({"update":update,"loss":round(float(loss),7),"regression":round(float(regression),7),"gate":round(float(gate),7),"gradient":round(gradient,7)})
        raw={name:value.detach().cpu() for name,value in model.state_dict().items()};smooth={name:value.detach().cpu() for name,value in ema.state_dict().items()};payload={"format":CHECKPOINT_FORMAT,"source_sha256":source_sha256(),"corpus_sha256":manifest["manifest_sha256"],"update":end,"model_state":raw,"ema_state":smooth,"model_state_sha256":state_sha256(raw),"ema_state_sha256":state_sha256(smooth),"optimizer_state":optimizer.state_dict(),"rng_state":rng.bit_generator.state,"history":history,"normalization":{"mean":mean.tolist(),"std":std.tolist()},"plan":plan.to_dict(),"runtime":{"seconds":round(time.perf_counter()-began,6),"peak_reserved_bytes":int(torch.cuda.max_memory_reserved(device))}};temporary=output/f".actor-{end}.tmp";torch.save(payload,temporary);os.replace(temporary,output/f"actor_{end:07d}.pt");print(json.dumps({"update":end,"loss":history[-1]["loss"],**payload["runtime"]}),flush=True)
    return {"passed":True,"checkpoint":str(output/f"actor_{plan.updates:07d}.pt"),"ema_state_sha256":payload["ema_state_sha256"]}
@torch.inference_mode()
def _predict(model,episode,payload,device,threshold,alpha):
    mn=torch.tensor(payload["normalization"]["mean"],device=device);sd=torch.tensor(payload["normalization"]["std"],device=device);current=(torch.from_numpy(episode["actor_state"]).to(device)-mn)/sd;previous_raw=np.concatenate((episode["actor_state"][:1],episode["actor_state"][:-1]));previous=(torch.from_numpy(previous_raw).to(device)-mn)/sd;action=torch.from_numpy(episode["action"].astype(np.int64)).to(device);control=torch.from_numpy(episode["control"]).to(device);state=torch.from_numpy(episode["state"]).to(device);result=model(current,previous,action,control,state);keep=result.gate>=threshold;prediction=(current+alpha*keep*(result.state-current))*sd+mn;wrong=model(current,previous,(action+7)%model.actions,control,state);wrong_prediction=(current+alpha*(wrong.gate>=threshold)*(wrong.state-current))*sd+mn;return prediction.cpu(),wrong_prediction.cpu()
def evaluate(output:Path=DEFAULT_OUTPUT,*,corpus:Path=DEFAULT_CORPUS):
    output=Path(output).resolve();episodes,manifest=load_encoded_corpus(corpus);validation,test=episodes[4],episodes[5];candidates=[];device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for checkpoint in sorted(output.glob("actor_*.pt")):
        payload=torch.load(checkpoint,map_location="cpu",weights_only=True)
        for variant,key in (("raw","model_state"),("ema","ema_state")):
            model=ActorStateStudent();model.load_state_dict(payload[key]);model.to(device).eval()
            for threshold in (0,.1,.2,.3,.4,.5,.6,.7,.8,.9):
                for alpha in (.05,.1,.15,.2,.3,.4,.5,.6,.7,.8,.9,1.):
                    pred,wrong=_predict(model,validation,payload,device,threshold,alpha);target=torch.from_numpy(validation["target_actor_state"]);current=torch.from_numpy(validation["actor_state"]);p=float(F.l1_loss(current,target));mae=float(F.l1_loss(pred,target));candidates.append({"checkpoint":checkpoint.name,"variant":variant,"threshold":threshold,"alpha":alpha,"improvement":1-mae/p,"wrong_advantage":float(F.l1_loss(wrong,target)-mae)})
    valid=[row for row in candidates if row["improvement"]>0 and row["wrong_advantage"]>0];chosen=max(valid,key=lambda row:row["improvement"]+row["wrong_advantage"]) if valid else max(candidates,key=lambda row:row["improvement"]);checkpoint=output/chosen["checkpoint"];payload=torch.load(checkpoint,map_location="cpu",weights_only=True);model=ActorStateStudent();model.load_state_dict(payload["model_state" if chosen["variant"]=="raw" else "ema_state"]);model.to(device).eval();pred,wrong=_predict(model,test,payload,device,chosen["threshold"],chosen["alpha"]);target=torch.from_numpy(test["target_actor_state"]);current=torch.from_numpy(test["actor_state"]);p=float(F.l1_loss(current,target));mae=float(F.l1_loss(pred,target));metrics={"mae":mae,"persistence_mae":p,"improvement":1-mae/p,"correct_action_advantage":float(F.l1_loss(wrong,target)-mae)};gates={"beats_persistence":metrics["improvement"]>0,"correct_beats_wrong_action":metrics["correct_action_advantage"]>0};gates["all_passed"]=all(gates.values());report={"format":REPORT_FORMAT,"status":"ready" if gates["all_passed"] else "experimental","source_sha256":source_sha256(),"corpus_sha256":manifest["manifest_sha256"],"parameters":model.parameter_count,"selection":chosen,"test":metrics,"gates":gates,"checkpoint":{"path":checkpoint.name,"bytes":checkpoint.stat().st_size,"sha256":_hash(checkpoint)}};report["report_sha256"]=hashlib.sha256(canonical(report)).hexdigest();(output/"report.json").write_bytes(canonical(report));return report
