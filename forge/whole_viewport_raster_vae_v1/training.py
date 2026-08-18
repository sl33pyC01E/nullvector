from __future__ import annotations

import copy,hashlib,json,math,os,time,uuid
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import torch
from torch.nn import functional as F

from ..action_teacher_viewport_v5 import validate_trajectory
from ..recurrent_world_pipeline_v1.contract import DECODER,file_sha256
from ..world_frame_vae.contract import ModelConfig
from ..world_frame_vae.model import WorldFrameVAE
from .contract import DEFAULT_CORPUS,DEFAULT_OUTPUT,FORMAT,TrainingPlan,canonical,plan_dict,source_sha256

def _atomic(payload,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}");torch.save(payload,temporary);os.replace(temporary,path)

def _decoder_parameters(model):return [parameter for module in (model.from_latent,model.decoder,model.out) for parameter in module.parameters()]

def _loss(prediction,target):
    difference=prediction-target;l1=difference.abs().mean();mse=difference.square().mean();pdx,tdx=prediction[:,:,:,1:]-prediction[:,:,:,:-1],target[:,:,:,1:]-target[:,:,:,:-1];pdy,tdy=prediction[:,:,1:]-prediction[:,:,:-1],target[:,:,1:]-target[:,:,:-1];edge=F.l1_loss(pdx,tdx)+F.l1_loss(pdy,tdy);high=F.l1_loss(prediction-F.avg_pool2d(prediction,3,1,1),target-F.avg_pool2d(target,3,1,1));coarse=F.l1_loss(F.avg_pool2d(prediction,4),F.avg_pool2d(target,4));return l1*7+mse*2+edge*6+high*4+coarse,(l1,edge,high)

@torch.inference_mode()
def _encode(model,frames,device,batch=8):
    result=[]
    for start in range(0,len(frames),batch):
        tensor=torch.as_tensor(frames[start:start+batch],device=device).permute(0,3,1,2).float()/255
        with torch.autocast("cuda",dtype=torch.bfloat16):mean,_=model.encode(tensor)
        result.append(mean.half().cpu().numpy())
    return np.concatenate(result)

@torch.inference_mode()
def _metrics(model,latents,frames,device,batch=8,collect=False):
    mae=mse=edge=0.;images=[]
    for start in range(0,len(frames),batch):
        stop=min(len(frames),start+batch);target=torch.as_tensor(frames[start:stop],device=device).permute(0,3,1,2).float()/255
        with torch.autocast("cuda",dtype=torch.bfloat16):prediction=model.decode(torch.as_tensor(latents[start:stop],device=device).float()).float().clamp(0,1)
        mae+=float(F.l1_loss(prediction,target))*len(target);mse+=float(F.mse_loss(prediction,target))*len(target);pdx,tdx=prediction[:,:,:,1:]-prediction[:,:,:,:-1],target[:,:,:,1:]-target[:,:,:,:-1];pdy,tdy=prediction[:,:,1:]-prediction[:,:,:-1],target[:,:,1:]-target[:,:,:-1];edge+=float(F.l1_loss(pdx,tdx)+F.l1_loss(pdy,tdy))*len(target)
        if collect:images.append((prediction.permute(0,2,3,1).cpu().numpy()*255+.5).astype(np.uint8))
    metrics={"mae":mae/len(frames),"mse":mse/len(frames),"psnr_db":-10*math.log10(max(mse/len(frames),1e-12)),"edge_mae":edge/len(frames)}
    return metrics,(np.concatenate(images) if images else None)

def _contact(path,frames,reconstructions):
    indices=np.linspace(0,len(frames)-1,8).round().astype(int);sheet=Image.new("RGB",(512,8*276),(4,10,13));draw=ImageDraw.Draw(sheet)
    for row,index in enumerate(indices):sheet.paste(Image.fromarray(frames[index]),(0,row*276+20));sheet.paste(Image.fromarray(reconstructions[index]),(256,row*276+20));draw.text((8,row*276+4),f"TEACHER F{index:04}",fill=(112,231,224));draw.text((264,row*276+4),"ADAPTED VAE RECONSTRUCTION",fill=(255,104,192))
    sheet.save(path,optimize=True)

def _load_frames(root):
    episodes=[];manifests=[]
    for path in sorted(Path(root).glob("*/manifest.json")):
        manifest=validate_trajectory(path.parent)
        with np.load(path.parent/manifest["artifact"]["path"],allow_pickle=False) as archive:episodes.append(archive["frame"])
        manifests.append(manifest["manifest_sha256"])
    if len(episodes)!=30 or any(len(item)!=384 for item in episodes):raise ValueError("whole-viewport VAE corpus balance drifted")
    heldout={0,7,14,21,28};train=np.concatenate([item for index,item in enumerate(episodes) if index not in heldout]);validation=np.concatenate([item for index,item in enumerate(episodes) if index in heldout]);return train,validation,manifests

def train(*,corpus:Path=DEFAULT_CORPUS,output:Path=DEFAULT_OUTPUT,plan=TrainingPlan()):
    output=Path(output);parent_sha=file_sha256(DECODER)
    if output.exists():raise FileExistsError(output)
    if not torch.cuda.is_available():raise RuntimeError("whole-viewport VAE requires CUDA")
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.52,0);torch.manual_seed(plan.seed);rng=np.random.default_rng(plan.seed);device=torch.device("cuda")
    train_frames,validation_frames,manifests=_load_frames(corpus);payload=torch.load(DECODER,map_location="cpu",weights_only=True);parent=WorldFrameVAE(ModelConfig(**payload["model_config"]));parent.load_state_dict(payload["state"]);parent.to(device).eval().requires_grad_(False);train_latents=_encode(parent,train_frames,device);validation_latents=_encode(parent,validation_frames,device)
    model=copy.deepcopy(parent).train().requires_grad_(False);parameters=_decoder_parameters(model)
    for parameter in parameters:parameter.requires_grad_(True)
    ema=copy.deepcopy(model).eval().requires_grad_(False);optimizer=torch.optim.AdamW(parameters,lr=plan.learning_rate,weight_decay=1e-4,fused=True);work=output.parent/f".{output.name}.work";latest=work/"latest.pt";history=[];start=0;checkpoint=None
    identity={"format":FORMAT,"source_sha256":source_sha256(),"parent_sha256":parent_sha,"corpus_manifests":manifests,"plan":plan_dict(plan)}
    if latest.exists():
        resume=torch.load(latest,map_location=device,weights_only=False)
        if any(resume.get(key)!=value for key,value in identity.items()):raise ValueError("whole-viewport VAE resume drifted")
        model.load_state_dict(resume["model_state"]);ema.load_state_dict(resume["ema_state"]);optimizer.load_state_dict(resume["optimizer_state"]);rng.bit_generator.state=resume["rng_state"];history=list(resume["history"]);start=int(resume["update"]);checkpoint=resume
    for segment_end in range(start+plan.segment_updates,plan.updates+1,plan.segment_updates):
        began=time.perf_counter();torch.cuda.reset_peak_memory_stats(device)
        for update in range(segment_end-plan.segment_updates+1,segment_end+1):
            indices=rng.integers(0,len(train_frames),plan.batch_size);latent=torch.as_tensor(train_latents[indices],device=device).float();target=torch.as_tensor(train_frames[indices],device=device).permute(0,3,1,2).float()/255;optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.bfloat16):prediction=model.decode(latent);loss,parts=_loss(prediction,target)
            loss.backward();gradient=float(torch.nn.utils.clip_grad_norm_(parameters,1));optimizer.step()
            with torch.no_grad():
                for smooth,current in zip(_decoder_parameters(ema),parameters):smooth.lerp_(current,1-plan.ema_decay)
            if update==1 or update%25==0:history.append({"update":update,"loss":float(loss),"l1":float(parts[0]),"edge":float(parts[1]),"high":float(parts[2]),"gradient":gradient})
        checkpoint={**identity,"update":segment_end,"model_state":{k:v.detach().cpu() for k,v in model.state_dict().items()},"ema_state":{k:v.detach().cpu() for k,v in ema.state_dict().items()},"optimizer_state":optimizer.state_dict(),"rng_state":rng.bit_generator.state,"history":history,"runtime":{"segment_seconds":time.perf_counter()-began,"peak_reserved_bytes":int(torch.cuda.max_memory_reserved(device))}};_atomic(checkpoint,latest);print(json.dumps({"update":segment_end,"history":history[-1],"runtime":checkpoint["runtime"]}),flush=True)
    parent_metrics,_=_metrics(parent,validation_latents,validation_frames,device);raw_metrics,_=_metrics(model.eval(),validation_latents,validation_frames,device);ema_metrics,_=_metrics(ema.eval(),validation_latents,validation_frames,device);selected_name,selected,selected_metrics=("raw",model,raw_metrics) if raw_metrics["mae"]<=ema_metrics["mae"] else ("ema",ema,ema_metrics);selected_state={name:value.detach().cpu() for name,value in selected.state_dict().items()};improvements={key:1-selected_metrics[key]/parent_metrics[key] for key in ("mae","mse","edge_mae")};gates={"mae_improves_5pct":improvements["mae"]>.05,"edge_improves":improvements["edge_mae"]>0,"under_13gib_vram":checkpoint["runtime"]["peak_reserved_bytes"]<13*1024**3};gates["all_passed"]=all(gates.values());manifest={**identity,"status":"accepted" if gates["all_passed"] else "rejected","model_config":payload["model_config"],"selection":selected_name,"validation":{"parent":parent_metrics,"raw":raw_metrics,"ema":ema_metrics,"selected":selected_metrics,"improvements":improvements},"gates":gates,"history":history,"runtime":checkpoint["runtime"]};runtime_payload={**manifest,"state":selected_state};release_path=output.parent/f".{output.name}.tmp-{uuid.uuid4().hex}";release_path.mkdir(parents=True);_atomic(runtime_payload,release_path/"runtime.pt");manifest["artifact"]={"path":"runtime.pt","bytes":(release_path/"runtime.pt").stat().st_size,"sha256":file_sha256(release_path/"runtime.pt")};manifest["manifest_sha256"]=hashlib.sha256(canonical(manifest)).hexdigest();(release_path/"manifest.json").write_bytes(canonical(manifest));_,images=_metrics(selected,validation_latents,validation_frames,device,collect=True);_contact(release_path/"contact.png",validation_frames,images);os.replace(release_path,output);return manifest
