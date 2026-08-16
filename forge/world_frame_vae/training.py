from __future__ import annotations
import argparse,hashlib,json,math,random
from pathlib import Path
import numpy as np,torch
from PIL import Image,ImageDraw
from torch.nn import functional as F
from .contract import CHECKPOINT_FORMAT,ModelConfig,TrainingConfig,canonical,config_dict,source_sha256
from .data import load_episodes
from .model import WorldFrameVAE

def _hash(state):
    digest=hashlib.sha256()
    for name,value in sorted(state.items()):digest.update(name.encode()+b"\0"+value.cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
def _loss(model,batch,kl_weight):
    reconstruction,mean,logvar=model(batch);l1=F.l1_loss(reconstruction,batch);mse=F.mse_loss(reconstruction,batch);dx=reconstruction[:,:,:,1:]-reconstruction[:,:,:,:-1];target_dx=batch[:,:,:,1:]-batch[:,:,:,:-1];dy=reconstruction[:,:,1:]-reconstruction[:,:,:-1];target_dy=batch[:,:,1:]-batch[:,:,:-1];edge=F.l1_loss(dx,target_dx)+F.l1_loss(dy,target_dy);laplace=F.l1_loss(dx[:,:,:,1:]-dx[:,:,:,:-1],target_dx[:,:,:,1:]-target_dx[:,:,:,:-1])+F.l1_loss(dy[:,:,1:]-dy[:,:,:-1],target_dy[:,:,1:]-target_dy[:,:,:-1]);multi=F.l1_loss(F.avg_pool2d(reconstruction,2),F.avg_pool2d(batch,2));kl=(-.5*(1+logvar-mean.square()-logvar.exp())).mean();return l1*5+mse*2+edge*3+laplace*1.2+multi*.6+kl*kl_weight,(l1,mse,edge,kl)
def _contact(path,original,reconstruction):
    count=min(8,len(original));sheet=Image.new("RGB",(256*count,256*2),(0,0,0))
    for index in range(count):sheet.paste(Image.fromarray(original[index]),(index*256,0));sheet.paste(Image.fromarray(np.clip(reconstruction[index]*255,0,255).astype(np.uint8)),(index*256,256))
    sheet.resize((256*count,1024),Image.Resampling.NEAREST).save(path)
def train(output:Path,episodes,*,config=ModelConfig(),training=TrainingConfig()):
    torch.manual_seed(training.seed);np.random.seed(training.seed&0xffffffff);random.seed(training.seed);torch.backends.cuda.matmul.allow_tf32=True;device=torch.device("cuda" if torch.cuda.is_available() else "cpu");frames,sources,corpus_sha=load_episodes(episodes);split=max(1,len(frames)-60);train_tensor=torch.from_numpy(frames[:split]).permute(0,3,1,2).float().div_(255);validation=torch.from_numpy(frames[split:]).permute(0,3,1,2).float().div_(255);model=WorldFrameVAE(config).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=training.learning_rate,weight_decay=1e-4,fused=device.type=="cuda");ema={name:value.detach().clone() for name,value in model.state_dict().items()};rng=np.random.default_rng(training.seed);history=[]
    for step in range(1,training.steps+1):
        indices=torch.from_numpy(rng.integers(0,len(train_tensor),training.batch_size));batch=train_tensor[indices].to(device);optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):loss,parts=_loss(model,batch,training.kl_weight)
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
        with torch.no_grad():
            for name,value in model.state_dict().items():ema[name].lerp_(value.detach(),1-training.ema_decay)
        if step==1 or step%200==0 or step==training.steps:history.append({"step":step,"loss":round(float(loss),6),"l1":round(float(parts[0]),6),"edge":round(float(parts[2]),6),"kl":round(float(parts[3]),6)})
    ema_cpu={name:value.detach().cpu() for name,value in ema.items()};model.load_state_dict(ema_cpu);model.to(device).eval();reconstructions=[]
    with torch.inference_mode():
        for start in range(0,len(validation),8):reconstruction,_,_=model(validation[start:start+8].to(device),sample=False);reconstructions.append(reconstruction.float().cpu())
    reconstructed=torch.cat(reconstructions);mae=float(F.l1_loss(reconstructed,validation));mse=float(F.mse_loss(reconstructed,validation));psnr=-10*math.log10(max(mse,1e-12));edge=float(F.l1_loss(reconstructed[:,:,:,1:]-reconstructed[:,:,:,:-1],validation[:,:,:,1:]-validation[:,:,:,:-1])+F.l1_loss(reconstructed[:,:,1:]-reconstructed[:,:,:-1],validation[:,:,1:]-validation[:,:,:-1]));parameters=sum(parameter.numel() for parameter in model.parameters());report={"format":"nullvector-world-frame-vae-training/1.0.0","source_sha256":source_sha256(),"corpus_sha256":corpus_sha,"sources":list(sources),"parameters":parameters,"device":str(device),"steps":training.steps,"train_frames":len(train_tensor),"heldout_frames":len(validation),"heldout_mae":mae,"heldout_mse":mse,"heldout_psnr_db":psnr,"heldout_edge_mae":edge,"latent_shape":[config.latent_channels,32,32],"history":history};payload={"format":CHECKPOINT_FORMAT,"source_sha256":source_sha256(),"corpus_sha256":corpus_sha,"model_config":config_dict(config),"training_config":config_dict(training),"ema_state":ema_cpu,"ema_sha256":_hash(ema_cpu),"report":report};output.mkdir(parents=True,exist_ok=False);torch.save(payload,output/"checkpoint.pt");(output/"report.json").write_bytes(canonical(report));_contact(output/"heldout_contact_sheet.png",frames[split:],reconstructed.permute(0,2,3,1).numpy());return report
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,required=True);parser.add_argument("--episodes",type=Path,nargs="+",required=True);parser.add_argument("--steps",type=int,default=4800);parser.add_argument("--batch-size",type=int,default=8);args=parser.parse_args();print(json.dumps(train(args.output,args.episodes,training=TrainingConfig(steps=args.steps,batch_size=args.batch_size)),indent=2))
