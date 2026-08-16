from __future__ import annotations
import argparse,hashlib,json,math,random
from pathlib import Path
import numpy as np,torch
from PIL import Image
from torch.nn import functional as F
from .contract import CHECKPOINT_FORMAT,ModelConfig,TrainingConfig,canonical,config_dict,source_sha256
from .data import encode_episodes
from .model import ActionDiT
from .runtime import WorldActionDiTRuntime
from ..world_frame_vae import WorldFrameVAERuntime

def _hash(state):
    digest=hashlib.sha256()
    for name,value in sorted(state.items()):digest.update(name.encode()+b"\0"+value.cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
def _concat(episodes,name):return np.concatenate([episode[name] for episode in episodes])
def _contact(path,current,target,predicted):
    count=min(8,len(current));sheet=Image.new("RGB",(256*count,256*3))
    for index in range(count):sheet.paste(Image.fromarray(current[index]),(index*256,0));sheet.paste(Image.fromarray(target[index]),(index*256,256));sheet.paste(Image.fromarray(predicted[index]),(index*256,512))
    sheet.resize((256*count,1536),Image.Resampling.NEAREST).save(path)
def train(output:Path,episodes,vae_checkpoint:Path,*,config=ModelConfig(),training=TrainingConfig()):
    output.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(training.seed);np.random.seed(training.seed&0xffffffff);random.seed(training.seed);torch.backends.cuda.matmul.allow_tf32=True;device=torch.device("cuda" if torch.cuda.is_available() else "cpu");vae=WorldFrameVAERuntime.from_checkpoint(vae_checkpoint,device=str(device));encoded,sources,corpus_sha=encode_episodes(episodes,vae,horizon=training.horizon);train_episodes=encoded[:-1];heldout=encoded[-1];current=_concat(train_episodes,"current");target=_concat(train_episodes,"target");control=_concat(train_episodes,"control");action=_concat(train_episodes,"action");state=_concat(train_episodes,"state");mean=np.mean(np.concatenate((current,target)),axis=(0,2,3));std=np.std(np.concatenate((current,target)),axis=(0,2,3))+1e-4;current=(current-mean[None,:,None,None])/std[None,:,None,None];target=(target-mean[None,:,None,None])/std[None,:,None,None];model=ActionDiT(config).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=training.learning_rate,weight_decay=2e-3,fused=device.type=="cuda");ema={name:value.detach().clone() for name,value in model.state_dict().items()};rng=np.random.default_rng(training.seed);history=[]
    for step in range(1,training.steps+1):
        indices=rng.integers(0,len(current),training.batch_size);x0=torch.from_numpy(current[indices]).to(device);x1=torch.from_numpy(target[indices]).to(device);t=torch.rand(training.batch_size,device=device);interpolated=x0+(x1-x0)*t[:,None,None,None]+torch.randn_like(x0)*(.025*torch.sin(t*math.pi))[:,None,None,None];velocity=x1-x0;act=torch.from_numpy(action[indices].astype(np.int64)).to(device);ctl=torch.from_numpy(control[indices]).to(device);st=torch.from_numpy(state[indices]).to(device);optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):prediction=model(interpolated,t,act,ctl,st);base=F.smooth_l1_loss(prediction,velocity);edge=F.l1_loss(prediction[:,:,:,1:]-prediction[:,:,:,:-1],velocity[:,:,:,1:]-velocity[:,:,:,:-1])+F.l1_loss(prediction[:,:,1:]-prediction[:,:,:-1],velocity[:,:,1:]-velocity[:,:,:-1]);loss=base+edge*.25
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
        with torch.no_grad():
            for name,value in model.state_dict().items():ema[name].lerp_(value.detach(),1-training.ema_decay)
        if step==1 or step%250==0 or step==training.steps:history.append({"step":step,"loss":round(float(loss),6),"velocity":round(float(base),6),"edge":round(float(edge),6)})
    ema_cpu={name:value.detach().cpu() for name,value in ema.items()}
    recovery_payload={"format":CHECKPOINT_FORMAT,"source_sha256":source_sha256(),"corpus_sha256":corpus_sha,"model_config":config_dict(config),"training_config":config_dict(training),"latent_mean":mean.tolist(),"latent_std":std.tolist(),"ema_state":ema_cpu,"ema_sha256":_hash(ema_cpu),"history":history,"status":"trained_pending_evaluation"}
    torch.save(recovery_payload,output/"trained_pending_evaluation.pt")
    model.load_state_dict(ema_cpu);model.to(device).eval();mean_tensor=torch.from_numpy(mean).to(device).view(1,-1,1,1);std_tensor=torch.from_numpy(std).to(device).view(1,-1,1,1);held_current=torch.from_numpy(heldout["current"]).to(device);held_target=torch.from_numpy(heldout["target"]).to(device);normalized=(held_current-mean_tensor)/std_tensor;runtime=WorldActionDiTRuntime(model,device,{},mean_tensor,std_tensor);predicted=[]
    for start in range(0,len(normalized),16):predicted.append(runtime.predict_latent(held_current[start:start+16],action=heldout["action"][start:start+16],control=heldout["control"][start:start+16],state=heldout["state"][start:start+16],steps=8).cpu())
    predicted=torch.cat(predicted);held_current_cpu=held_current.cpu();held_target_cpu=held_target.cpu();persistence_latent=float(F.l1_loss(held_current_cpu,held_target_cpu));model_latent=float(F.l1_loss(predicted,held_target_cpu));decoded=[]
    with torch.inference_mode():
        for start in range(0,len(predicted),8):decoded.append(vae.model.decode(predicted[start:start+8].to(device)).float().cpu())
    decoded=torch.cat(decoded);target_frame=torch.from_numpy(heldout["target_frame"]).permute(0,3,1,2).float().div_(255);current_frame=torch.from_numpy(heldout["current_frame"]).permute(0,3,1,2).float().div_(255);model_rgb=float(F.l1_loss(decoded,target_frame));persistence_rgb=float(F.l1_loss(current_frame,target_frame));predicted_frame=np.clip(decoded.permute(0,2,3,1).numpy()*255,0,255).astype(np.uint8);report={"format":"nullvector-world-latent-action-dit-training/1.0.0","source_sha256":source_sha256(),"corpus_sha256":corpus_sha,"vae_source_sha256":vae.report["source_sha256"],"sources":list(sources),"parameters":sum(parameter.numel() for parameter in model.parameters()),"device":str(device),"steps":training.steps,"horizon":training.horizon,"train_pairs":len(current),"heldout_pairs":len(held_current),"heldout_model_latent_mae":model_latent,"heldout_persistence_latent_mae":persistence_latent,"heldout_model_rgb_mae":model_rgb,"heldout_persistence_rgb_mae":persistence_rgb,"latent_improvement":1-model_latent/persistence_latent,"rgb_improvement":1-model_rgb/persistence_rgb,"history":history};payload={"format":CHECKPOINT_FORMAT,"source_sha256":source_sha256(),"corpus_sha256":corpus_sha,"model_config":config_dict(config),"training_config":config_dict(training),"latent_mean":mean.tolist(),"latent_std":std.tolist(),"ema_state":ema_cpu,"ema_sha256":_hash(ema_cpu),"report":report};torch.save(payload,output/"checkpoint.pt");(output/"report.json").write_bytes(canonical(report));_contact(output/"heldout_contact_sheet.png",heldout["current_frame"],heldout["target_frame"],predicted_frame);return report
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,required=True);parser.add_argument("--episodes",type=Path,nargs="+",required=True);parser.add_argument("--vae",type=Path,required=True);parser.add_argument("--steps",type=int,default=5000);parser.add_argument("--batch-size",type=int,default=32);args=parser.parse_args();print(json.dumps(train(args.output,args.episodes,args.vae,training=TrainingConfig(steps=args.steps,batch_size=args.batch_size)),indent=2))
