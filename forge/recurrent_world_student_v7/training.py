from __future__ import annotations
import copy,json,os,time
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from ..recurrent_world_student_v5.model import PerceptionRecurrentWorldStudent
from ..recurrent_world_student_v6.calibration import _score
from ..recurrent_world_student_v6.training import _normalizers,_sample_batch
from ..safety import require_disk_floor
from ..world_action_natural_v10 import load
from ..world_frame_decoder_adapt_v1 import AdaptedWorldFrameCodec
from ..world_latent_dit.contract import ModelConfig
from .contract import CHECKPOINT_FORMAT,CODEC,CODEC_SHA256,CORPUS,DEFAULT_OUTPUT,PARENT,PARENT_SHA256,TrainingPlan,file_sha256,source_sha256

def _atomic(path,payload):
    temporary=path.with_name(f".{path.name}.tmp-{os.getpid()}");torch.save(payload,temporary);os.replace(temporary,path)

def _frame_targets(rows,steps,device,count):
    values=np.stack([sequence["frame"][start+steps] for sequence,start in rows[:count]]);return torch.from_numpy(values).permute(0,3,1,2).float().div_(255).to(device)

def _rows(sequences,rng,count,steps):
    result=[]
    for _ in range(count):
        sequence=sequences[int(rng.integers(0,len(sequences)))];start=int(rng.integers(1,len(sequence["latent"])-steps));result.append((sequence,start))
    return result

def _batch_from_rows(rows,steps,device):
    def gather(name,begin,length):return torch.from_numpy(np.stack([sequence[name][start+begin:start+begin+length] for sequence,start in rows])).to(device,non_blocking=True)
    return {"latent":gather("latent",-1,steps+2),"actor":gather("actor_state",-1,steps+2),"action":gather("action",1,steps).long(),"control":gather("control",1,steps),"state":gather("state",0,steps),"visibility":gather("visibility",0,steps),"memory":gather("memory",0,steps)}

@torch.inference_mode()
def _pixel_metrics(model,codec,sequence,norms,device,horizon,bias,ramp,samples=16):
    lm,ls,am,ass=norms;starts=np.linspace(1,len(sequence["latent"])-horizon-1,min(samples,len(sequence["latent"])-horizon-1),dtype=np.int64);previous=torch.from_numpy(sequence["latent"][starts-1]).to(device);current=torch.from_numpy(sequence["latent"][starts]).to(device);initial=current.clone();previous_actor=torch.from_numpy(sequence["actor_state"][starts-1]).to(device);actor=torch.from_numpy(sequence["actor_state"][starts]).to(device)
    for offset in range(horizon):
        ix=starts+offset;action=torch.from_numpy(sequence["action"][ix+1].astype(np.int64)).to(device);control=torch.from_numpy(sequence["control"][ix+1]).to(device);state=torch.from_numpy(sequence["state"][ix]).to(device);visibility=torch.from_numpy(sequence["visibility"][ix]).to(device);memory=torch.from_numpy(sequence["memory"][ix]).to(device);cn,pn=(current-lm)/ls,(previous-lm)/ls;delta,logits=model.gated_action(cn,pn,action,control,state,actor,visibility,memory);applied=bias*min(offset/ramp,1.) if ramp>0 else bias;next_latent=(cn+torch.sigmoid(logits+applied)*delta)*ls+lm;an,pan=(actor-am)/ass,(previous_actor-am)/ass;result=model.actor(an,pan,action,control,state,visibility,memory);next_actor=(an+.9*(result.gate>=.7)*(result.state-an))*ass+am;previous,current=current,next_latent;previous_actor,actor=actor,next_actor
    prediction=torch.clamp(codec.model.decode(current),0,1);baseline=torch.from_numpy(sequence["frame"][starts]).permute(0,3,1,2).float().div_(255).to(device);target=torch.from_numpy(sequence["frame"][starts+horizon]).permute(0,3,1,2).float().div_(255).to(device);mae=float(F.l1_loss(prediction,target));persistence=float(F.l1_loss(baseline,target));motion=float(F.l1_loss(prediction,baseline));return {"horizon":horizon,"samples":len(starts),"mae":mae,"persistence_mae":persistence,"improvement":1-mae/persistence,"motion_ratio":motion/persistence}

def train(output:Path=DEFAULT_OUTPUT,*,plan:TrainingPlan=TrainingPlan()):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True);require_disk_floor(output.parent,floor_gb=100,planned_bytes=3*1024**3)
    if not torch.cuda.is_available():raise RuntimeError("decoder-aware V7 training requires CUDA")
    if file_sha256(PARENT)!=PARENT_SHA256 or file_sha256(CODEC)!=CODEC_SHA256:raise ValueError("decoder-aware V7 parent/codec drifted")
    if plan.total_updates%plan.segment_updates:raise ValueError("V7 segment plan invalid")
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.45,0);torch.manual_seed(plan.seed);rng=np.random.default_rng(plan.seed);device=torch.device("cuda:0");sequences,manifest=load(CORPUS);parent=torch.load(PARENT,map_location="cpu",weights_only=True);model=PerceptionRecurrentWorldStudent(ModelConfig(**parent["model_config"]));model.load_state_dict(parent["state"]);model.to(device);ema=copy.deepcopy(model).eval().requires_grad_(False);codec=AdaptedWorldFrameCodec.from_checkpoint(CODEC,device="cuda");codec.model.requires_grad_(False);optimizer=torch.optim.AdamW(model.parameters(),lr=plan.learning_rate,weight_decay=1e-3,fused=True);norms=_normalizers(parent,device);lm,ls,am,ass=norms;inference=parent["inference"];bias=float(inference["gate_logit_bias_max"]);ramp=int(inference["gate_logit_bias_ramp_steps"]);latest=output/"latest.pt";history=[];start_update=0
    if latest.is_file():
        payload=torch.load(latest,map_location="cpu",weights_only=True)
        if payload.get("format")!=CHECKPOINT_FORMAT or payload.get("source_sha256")!=source_sha256() or payload.get("corpus_sha256")!=manifest["manifest_sha256"] or payload.get("parent_sha256")!=PARENT_SHA256 or payload.get("plan")!=plan.to_dict():raise ValueError("decoder-aware V7 resume drifted")
        model.load_state_dict(payload["model_state"]);ema.load_state_dict(payload["ema_state"]);optimizer.load_state_dict(payload["optimizer_state"]);rng.bit_generator.state=payload["rng_state"];history=list(payload["history"]);start_update=payload["update"]
    payload=None
    for end in range(start_update+plan.segment_updates,plan.total_updates+1,plan.segment_updates):
        began=time.perf_counter();torch.cuda.reset_peak_memory_stats(device);model.train()
        for update in range(end-plan.segment_updates+1,end+1):
            rows=_rows(sequences[:4],rng,plan.batch_size,plan.rollout_steps);batch=_batch_from_rows(rows,plan.rollout_steps,device);pixel_count=min(plan.pixel_batch_size,plan.batch_size);frame_target=_frame_targets(rows,plan.rollout_steps,device,pixel_count);optimizer.zero_grad(set_to_none=True);previous=batch["latent"][:,0];current=batch["latent"][:,1];previous_actor=batch["actor"][:,0];actor=batch["actor"][:,1];latent_total=actor_total=0.
            for offset in range(plan.rollout_steps):
                target=batch["latent"][:,offset+2];target_actor=batch["actor"][:,offset+2];visibility=batch["visibility"][:,offset];memory=batch["memory"][:,offset];cn,pn=(current-lm)/ls,(previous-lm)/ls;an,pan,tan=(actor-am)/ass,(previous_actor-am)/ass
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    delta,logits=model.gated_action(cn,pn,batch["action"][:,offset],batch["control"][:,offset],batch["state"][:,offset],actor,visibility,memory);target_delta=(target-current)/ls;magnitude=target_delta.abs().mean(1,keepdim=True)
                    with torch.no_grad():proposal=delta.float();truth=target_delta.float();trust=torch.clamp((proposal*truth).sum(1,keepdim=True)/(proposal.square().sum(1,keepdim=True)+1e-6),0,1)
                    applied=bias*min(offset/ramp,1.) if ramp>0 else bias;gated=torch.sigmoid(logits+applied)*delta;weight=1+5*torch.clamp(magnitude/.35,0,2);transition=(F.smooth_l1_loss(gated,target_delta,reduction="none")*weight).mean();proposal_loss=(F.smooth_l1_loss(delta,target_delta,reduction="none")*weight).mean();gate_loss=F.smooth_l1_loss(torch.sigmoid(logits),trust);actor_result=model.actor(an,pan,batch["action"][:,offset],batch["control"][:,offset],batch["state"][:,offset],visibility,memory);changed=(tan-an).abs()>.025;actor_loss=(F.smooth_l1_loss(actor_result.state,tan,reduction="none")*(1+6*changed)).mean();latent_loss=transition+plan.proposal_weight*proposal_loss+plan.gate_weight*gate_loss;loss=(latent_loss+plan.actor_weight*actor_loss)/plan.rollout_steps
                loss.backward(retain_graph=offset==plan.rollout_steps-1);latent_total+=float(latent_loss);actor_total+=float(actor_loss);next_latent=(cn+torch.sigmoid(logits+applied)*delta)*ls+lm
                if offset==plan.rollout_steps-1:
                    with torch.autocast("cuda",dtype=torch.bfloat16):
                        decoded=codec.model.decode(next_latent[:pixel_count]);pixel=F.smooth_l1_loss(decoded.float(),frame_target);edge=F.l1_loss(decoded[:,:,:,1:]-decoded[:,:,:,:-1],frame_target[:,:,:,1:]-frame_target[:,:,:,:-1])+F.l1_loss(decoded[:,:,1:]-decoded[:,:,:-1],frame_target[:,:,1:]-frame_target[:,:,:-1]);visual_loss=plan.pixel_weight*pixel+plan.edge_weight*edge
                    visual_loss.backward()
                with torch.no_grad():next_actor=(an+.9*(actor_result.gate>=.7)*(actor_result.state-an))*ass+am
                previous,current=current.detach(),next_latent.detach();previous_actor,actor=actor.detach(),next_actor.detach()
            gradient=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1));optimizer.step()
            with torch.no_grad():torch._foreach_mul_(list(ema.parameters()),plan.ema_decay);torch._foreach_add_(list(ema.parameters()),list(model.parameters()),alpha=1-plan.ema_decay)
            if update%10==0:history.append({"update":update,"latent":round(latent_total/plan.rollout_steps,7),"actor":round(actor_total/plan.rollout_steps,7),"pixel":round(float(pixel),7),"edge":round(float(edge),7),"gradient":round(gradient,7)})
        latent_validation={str(h):_score(ema.eval(),sequences[4],norms,device,h,bias,ramp,32) for h in (1,4,8)};pixel_validation=_pixel_metrics(ema.eval(),codec,sequences[4],norms,device,8,bias,ramp);score=sum(row["mae"]/row["persistence_mae"] for row in latent_validation.values())+pixel_validation["mae"]/pixel_validation["persistence_mae"];state={n:v.detach().cpu() for n,v in model.state_dict().items()};ema_state={n:v.detach().cpu() for n,v in ema.state_dict().items()};elapsed=time.perf_counter()-began;payload={"format":CHECKPOINT_FORMAT,"source_sha256":source_sha256(),"corpus_sha256":manifest["manifest_sha256"],"parent_sha256":PARENT_SHA256,"codec_sha256":CODEC_SHA256,"model_config":parent["model_config"],"normalization":parent["normalization"],"inference":inference,"plan":plan.to_dict(),"update":end,"model_state":state,"ema_state":ema_state,"optimizer_state":optimizer.state_dict(),"rng_state":rng.bit_generator.state,"history":history,"validation":{"latent":latent_validation,"pixel":pixel_validation},"selection_score":score,"runtime":{"segment_seconds":round(elapsed,6),"updates_per_second":round(plan.segment_updates/elapsed,4),"peak_reserved_bytes":int(torch.cuda.max_memory_reserved(device))}};_atomic(latest,payload);_atomic(output/f"milestone_{end:07d}.pt",payload);print(json.dumps({"update":end,"validation":payload["validation"],"selection_score":score,"runtime":payload["runtime"]}),flush=True)
    return {"status":"trained_pending_decoder_aware_selection","update":payload["update"],"selection_score":payload["selection_score"],"runtime":payload["runtime"],"latest_sha256":file_sha256(latest)}
