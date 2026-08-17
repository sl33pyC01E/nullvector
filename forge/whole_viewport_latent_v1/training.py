from __future__ import annotations

import hashlib,json,os,random,shutil,uuid
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from ..recurrent_world_pipeline_v1.runtime import RecurrentWorldPipeline
from .contract import CHECKPOINT_FORMAT,DEFAULT_CORPUS,DEFAULT_OUTPUT,ModelConfig,TrainingConfig,canonical,config_dict,source_sha256
from .data import load_corpus,rows,sequence_starts
from .model import WholeViewportLatentModel

def _tensor(batch,name,device,dtype=torch.float32):return torch.as_tensor(batch[name],device=device,dtype=dtype)

def _targets(decoder,frame,device):
    image=torch.as_tensor(frame,device=device).permute(0,3,1,2).float()/255
    with torch.no_grad(),torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):latent=decoder.encode(image)[0]
    return image,latent.float()

def _atomic_torch_save(payload,path:Path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    torch.save(payload,temporary);os.replace(temporary,path)

@torch.inference_mode()
def _prepare_latents(decoder,data,device,batch_size=16):
    result=dict(data);current=[];previous=[]
    for start in range(0,len(data["frame"]),batch_size):
        stop=min(len(data["frame"]),start+batch_size)
        current.append(_targets(decoder,data["frame"][start:stop],device)[1].half().cpu().numpy())
        previous.append(_targets(decoder,data["previous_frame"][start:stop],device)[1].half().cpu().numpy())
    result["latent"]=np.concatenate(current);result["previous_latent"]=np.concatenate(previous);return result

@torch.inference_mode()
def evaluate(model,decoder,data,indices,device,batch_size=8):
    model.eval();latent_total=rgb_total=persistence_total=0.;changed_total=changed_persistence_total=stable_total=stable_persistence_total=0.;changed_pixels=stable_pixels=0.;count=0
    for start in range(0,len(indices),batch_size):
        chosen=indices[start:start+batch_size];batch={name:value[chosen] for name,value in data.items()};image=torch.as_tensor(batch["frame"],device=device).permute(0,3,1,2).float()/255;target=_tensor(batch,"latent",device);previous=_tensor(batch,"previous_latent",device)
        with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):pred=model(previous,_tensor(batch,"spatial",device),_tensor(batch,"organisms",device),_tensor(batch,"organism_mask",device,torch.bool),_tensor(batch,"state",device),_tensor(batch,"actor_state",device),_tensor(batch,"actor_field",device),_tensor(batch,"visibility",device),_tensor(batch,"memory",device),_tensor(batch,"control",device),_tensor(batch,"action",device,torch.long));decoded=decoder.decode(pred).float().clamp(0,1);persistence=decoder.decode(previous).float().clamp(0,1)
        n=len(chosen);latent_total+=F.l1_loss(pred.float(),target).item()*n;rgb_total+=F.l1_loss(decoded,image).item()*n;persistence_total+=F.l1_loss(persistence,image).item()*n
        changed=(image-persistence).abs().mean(1,keepdim=True)>(2/255);changed=F.max_pool2d(changed.float(),3,1,1).bool();stable=~changed
        pred_error=(decoded-image).abs().mean(1,keepdim=True);persistence_error=(persistence-image).abs().mean(1,keepdim=True)
        changed_total+=float(pred_error[changed].sum());changed_persistence_total+=float(persistence_error[changed].sum());changed_pixels+=int(changed.sum())
        stable_total+=float(pred_error[stable].sum());stable_persistence_total+=float(persistence_error[stable].sum());stable_pixels+=int(stable.sum());count+=n
    changed_mae=changed_total/max(changed_pixels,1);changed_persistence=changed_persistence_total/max(changed_pixels,1);stable_mae=stable_total/max(stable_pixels,1);stable_persistence=stable_persistence_total/max(stable_pixels,1)
    return {"latent_mae":latent_total/count,"rgb_mae":rgb_total/count,"persistence_rgb_mae":persistence_total/count,"rgb_improvement":1-rgb_total/max(persistence_total,1e-8),"changed_rgb_mae":changed_mae,"changed_persistence_rgb_mae":changed_persistence,"changed_rgb_improvement":1-changed_mae/max(changed_persistence,1e-8),"stable_rgb_mae":stable_mae,"stable_persistence_rgb_mae":stable_persistence,"stable_rgb_regression":stable_mae/max(stable_persistence,1e-8)-1,"changed_pixel_fraction":changed_pixels/max(changed_pixels+stable_pixels,1)}

@torch.inference_mode()
def evaluate_rollout(model,decoder,data,starts,device,horizon,batch_size=8):
    model.eval();latent_total=rgb_total=persistence_total=0.;count=0
    for begin in range(0,len(starts),batch_size):
        chosen=starts[begin:begin+batch_size];previous=_tensor({"value":data["previous_latent"][chosen]},"value",device)
        initial=previous
        target=None
        for offset in range(horizon):
            indices=chosen+offset
            previous=model(previous,_tensor({"value":data["spatial"][indices]},"value",device),_tensor({"value":data["organisms"][indices]},"value",device),_tensor({"value":data["organism_mask"][indices]},"value",device,torch.bool),_tensor({"value":data["state"][indices]},"value",device),_tensor({"value":data["actor_state"][indices]},"value",device),_tensor({"value":data["actor_field"][indices]},"value",device),_tensor({"value":data["visibility"][indices]},"value",device),_tensor({"value":data["memory"][indices]},"value",device),_tensor({"value":data["control"][indices]},"value",device),_tensor({"value":data["action"][indices]},"value",device,torch.long))
            target=_tensor({"value":data["latent"][indices]},"value",device)
        final_indices=chosen+horizon-1;image=torch.as_tensor(data["frame"][final_indices],device=device).permute(0,3,1,2).float()/255
        with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):
            decoded=decoder.decode(previous).float().clamp(0,1);persistence=decoder.decode(initial).float().clamp(0,1)
        n=len(chosen);latent_total+=F.l1_loss(previous.float(),target).item()*n;rgb_total+=F.l1_loss(decoded,image).item()*n;persistence_total+=F.l1_loss(persistence,image).item()*n;count+=n
    return {f"rollout_{horizon}_latent_mae":latent_total/max(count,1),f"rollout_{horizon}_rgb_mae":rgb_total/max(count,1),f"rollout_{horizon}_persistence_rgb_mae":persistence_total/max(count,1),f"rollout_{horizon}_rgb_improvement":1-rgb_total/max(persistence_total,1e-8)}

def train(*,corpus:Path=DEFAULT_CORPUS,output:Path=DEFAULT_OUTPUT,model_config=ModelConfig(),training=TrainingConfig(),device="cuda"):
    output=Path(output);corpus=Path(corpus)
    if output.exists():raise FileExistsError(output)
    episodes,manifests=load_corpus(corpus);count=sum(len(item["frame"]) for item in episodes)
    if count<64 or len(episodes)<3:raise ValueError("whole-viewport corpus needs at least three trajectories")
    target=torch.device(device if device=="cpu" or torch.cuda.is_available() else "cpu");torch.manual_seed(training.seed);np.random.seed(training.seed&0xffffffff);random.seed(training.seed)
    pipeline=RecurrentWorldPipeline.load(str(target));decoder=pipeline.decoder.eval()
    for parameter in decoder.parameters():parameter.requires_grad_(False)
    train_data=_prepare_latents(decoder,rows(episodes[:-1]),target);validation_data=_prepare_latents(decoder,rows(episodes[-1:]),target);train_indices=np.arange(len(train_data["frame"]));validation=np.arange(len(validation_data["frame"]));train_starts=sequence_starts(train_data,training.rollout_steps);validation_starts=sequence_starts(validation_data,training.rollout_steps)
    if not len(train_starts) or not len(validation_starts):raise ValueError("whole-viewport rollout corpus is incomplete")
    model=WholeViewportLatentModel(model_config).to(target);optimizer=torch.optim.AdamW(model.parameters(),lr=training.learning_rate,weight_decay=training.weight_decay);scaler=torch.amp.GradScaler("cuda",enabled=target.type=="cuda");ema={name:value.detach().clone() for name,value in model.state_dict().items()};rng=np.random.default_rng(training.seed);history=[];best=None;best_state=None;start_step=0
    work=output.parent/f".{output.name}.work";latest=work/"latest.pt";manifest_ids=[item["manifest_sha256"] for item in manifests]
    if latest.exists():
        resume=torch.load(latest,map_location=target,weights_only=False)
        expected={"source_sha256":source_sha256(),"model_config":config_dict(model_config),"training_config":config_dict(training),"corpus_manifests":manifest_ids}
        if any(resume.get(name)!=value for name,value in expected.items()):raise ValueError("whole-viewport resume provenance drifted")
        model.load_state_dict(resume["model"]);optimizer.load_state_dict(resume["optimizer"]);scaler.load_state_dict(resume["scaler"]);ema=resume["ema"];rng.bit_generator.state=resume["rng_state"];history=resume["history"];best=resume["best"];best_state=resume["best_state"];start_step=int(resume["step"])
    model.train()
    for step in range(start_step+1,training.steps+1):
        chosen=rng.choice(train_starts,size=min(training.batch_size,len(train_starts)),replace=len(train_starts)<training.batch_size);previous=_tensor({"value":train_data["previous_latent"][chosen]},"value",target);optimizer.zero_grad(set_to_none=True);latent_loss=rgb_loss=0.
        with torch.autocast(target.type,dtype=torch.bfloat16,enabled=target.type=="cuda"):
            for offset in range(training.rollout_steps):
                indices=chosen+offset;image=torch.as_tensor(train_data["frame"][indices],device=target).permute(0,3,1,2).float()/255;latent=_tensor({"value":train_data["latent"][indices]},"value",target);pred=model(previous,_tensor({"value":train_data["spatial"][indices]},"value",target),_tensor({"value":train_data["organisms"][indices]},"value",target),_tensor({"value":train_data["organism_mask"][indices]},"value",target,torch.bool),_tensor({"value":train_data["state"][indices]},"value",target),_tensor({"value":train_data["actor_state"][indices]},"value",target),_tensor({"value":train_data["actor_field"][indices]},"value",target),_tensor({"value":train_data["visibility"][indices]},"value",target),_tensor({"value":train_data["memory"][indices]},"value",target),_tensor({"value":train_data["control"][indices]},"value",target),_tensor({"value":train_data["action"][indices]},"value",target,torch.long));change=(latent-previous.detach()).abs();weight=1+4*change/change.mean(dim=(1,2,3),keepdim=True).clamp_min(.02);latent_loss=latent_loss+(F.smooth_l1_loss(pred.float(),latent,beta=.05,reduction="none")*weight).mean();decoded=decoder.decode(pred).float();previous_rgb=torch.as_tensor(train_data["previous_frame"][indices],device=target).permute(0,3,1,2).float()/255;pixel_change=(image-previous_rgb).abs().mean(1,keepdim=True);pixel_weight=1+4*pixel_change/pixel_change.mean(dim=(1,2,3),keepdim=True).clamp_min(1/255);rgb_loss=rgb_loss+(torch.abs(decoded-image)*pixel_weight).mean();previous=pred
            latent_loss=latent_loss/training.rollout_steps;rgb_loss=rgb_loss/training.rollout_steps;loss=latent_loss+training.rgb_weight*rgb_loss
        scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),1);scaler.step(optimizer);scaler.update()
        with torch.no_grad():
            decay=min(training.ema_decay,step/(step+10))
            for name,value in model.state_dict().items():ema[name].lerp_(value.detach(),1-decay)
        if step==1 or step%25==0:history.append({"step":step,"loss":float(loss),"latent":float(latent_loss),"rgb":float(rgb_loss)})
        if step%training.validation_every==0 or step==training.steps:
            raw={name:value.detach().clone() for name,value in model.state_dict().items()};model.load_state_dict(ema);metrics=evaluate(model,decoder,validation_data,validation,target,training.batch_size);metrics.update(evaluate_rollout(model,decoder,validation_data,validation_starts,target,training.rollout_steps,training.batch_size));model.load_state_dict(raw);score=metrics[f"rollout_{training.rollout_steps}_rgb_mae"]+metrics[f"rollout_{training.rollout_steps}_latent_mae"]*.1
            if best is None or score<best:best=score;best_state={name:value.detach().cpu().clone() for name,value in ema.items()}
            print(json.dumps({"step":step,**metrics,"best":best}),flush=True)
        if step%training.checkpoint_every==0 or step==training.steps:
            _atomic_torch_save({"source_sha256":source_sha256(),"model_config":config_dict(model_config),"training_config":config_dict(training),"corpus_manifests":manifest_ids,"step":step,"model":model.state_dict(),"optimizer":optimizer.state_dict(),"scaler":scaler.state_dict(),"ema":ema,"rng_state":rng.bit_generator.state,"history":history,"best":best,"best_state":best_state},latest)
    model.load_state_dict(best_state);final=evaluate(model,decoder,validation_data,validation,target,training.batch_size);final.update(evaluate_rollout(model,decoder,validation_data,validation_starts,target,training.rollout_steps,training.batch_size));staging=output.parent/f".{output.name}.tmp-{uuid.uuid4().hex}";staging.mkdir(parents=True)
    payload={"format":CHECKPOINT_FORMAT,"source_sha256":source_sha256(),"model_config":config_dict(model_config),"training_config":config_dict(training),"corpus_manifests":manifest_ids,"state":best_state,"validation":final,"parameters":sum(p.numel() for p in model.parameters())};checkpoint=staging/"model.pt";torch.save(payload,checkpoint);gates={"beats_rgb_persistence":final["rgb_improvement"]>0,"beats_changed_region_persistence":final["changed_rgb_improvement"]>0,"stable_region_regression_bounded":final["stable_rgb_regression"]<.10,f"beats_{training.rollout_steps}_step_persistence":final[f"rollout_{training.rollout_steps}_rgb_improvement"]>0};accepted=all(gates.values());manifest={"format":"nullvector-whole-viewport-latent-release/1.0.0","status":"accepted" if accepted else "rejected","source_sha256":source_sha256(),"frames":count,"episodes":len(manifests),"parameters":payload["parameters"],"validation":final,"gates":gates,"contract":{"runtime_graph":"previous visual latent + numeric ensemble tensors -> next visual latent -> full-frame VAE decode","traditional_world_graphics":False,"native_code_scope":["menus","hud","accessibility","debug"],"autoregressive_training_steps":training.rollout_steps},"artifact":{"path":checkpoint.name,"bytes":checkpoint.stat().st_size,"sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest()},"history":history};manifest["manifest_sha256"]=hashlib.sha256(canonical(manifest)).hexdigest();(staging/"manifest.json").write_bytes(canonical(manifest));os.replace(staging,output);return manifest
