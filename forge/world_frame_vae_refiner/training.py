from __future__ import annotations
import argparse,hashlib,json,math,random
from pathlib import Path
import numpy as np,torch
from PIL import Image
from torch.nn import functional as F
from ..world_frame_vae import WorldFrameVAERuntime
from ..world_frame_vae.data import load_episodes
from .contract import CHECKPOINT_FORMAT,ModelConfig,TrainingConfig,canonical,config_dict,source_sha256
from .model import PixelCellRefiner
from .runtime import file_sha256

def _hash(state):
    digest=hashlib.sha256()
    for name,value in sorted(state.items()):digest.update(name.encode()+b"\0"+value.cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()
def _edge(value):return value[:,:,:,1:]-value[:,:,:,:-1],value[:,:,1:]-value[:,:,:-1]
def _loss(prediction,target):
    dx,dy=_edge(prediction);tx,ty=_edge(target);blur=F.avg_pool2d(prediction,3,1,1);target_blur=F.avg_pool2d(target,3,1,1);l1=F.l1_loss(prediction,target);mse=F.mse_loss(prediction,target);edge=F.l1_loss(dx,tx)+F.l1_loss(dy,ty);high=F.l1_loss(prediction-blur,target-target_blur);laplace=F.l1_loss(dx[:,:,:,1:]-dx[:,:,:,:-1],tx[:,:,:,1:]-tx[:,:,:,:-1])+F.l1_loss(dy[:,:,1:]-dy[:,:,:-1],ty[:,:,1:]-ty[:,:,:-1]);return l1*6+mse+edge*4+high*3+laplace*1.5,(l1,mse,edge,high)
def _contact(path,original,base,refined):
    count=min(8,len(original));sheet=Image.new("RGB",(256*count,256*3))
    for index in range(count):
        for row,images in enumerate((original,base,refined)):sheet.paste(Image.fromarray(np.clip(images[index]*255,0,255).astype(np.uint8)),(index*256,row*256))
    sheet.save(path)

def train(output:Path,episodes,base_checkpoint:Path,*,config=ModelConfig(),training=TrainingConfig()):
    output.mkdir(parents=True,exist_ok=False);torch.manual_seed(training.seed);np.random.seed(training.seed&0xffffffff);random.seed(training.seed);torch.backends.cuda.matmul.allow_tf32=True;device=torch.device("cuda" if torch.cuda.is_available() else "cpu");frames,sources,corpus_sha=load_episodes(episodes);targets=torch.from_numpy(frames).permute(0,3,1,2).float().div_(255);split=max(1,len(targets)-60);base_runtime=WorldFrameVAERuntime.from_checkpoint(base_checkpoint,device=str(device));base_runtime.model.eval();base_rows=[]
    with torch.inference_mode():
        for start in range(0,len(targets),8):mean,_=base_runtime.model.encode(targets[start:start+8].to(device));base_rows.append(base_runtime.model.decode(mean).float().cpu())
    bases=torch.cat(base_rows);refiner=PixelCellRefiner(config).to(device);optimizer=torch.optim.AdamW(refiner.parameters(),lr=training.learning_rate,weight_decay=1e-4,fused=device.type=="cuda");ema={name:value.detach().clone() for name,value in refiner.state_dict().items()};rng=np.random.default_rng(training.seed);history=[]
    for step in range(1,training.steps+1):
        indices=rng.integers(0,split,training.batch_size);ys=rng.integers(0,257-training.crop,training.batch_size);xs=rng.integers(0,257-training.crop,training.batch_size);base=torch.stack([bases[i,:,y:y+training.crop,x:x+training.crop] for i,y,x in zip(indices,ys,xs)]).to(device);target=torch.stack([targets[i,:,y:y+training.crop,x:x+training.crop] for i,y,x in zip(indices,ys,xs)]).to(device);flip=torch.as_tensor(rng.random(training.batch_size)<.5,device=device)
        if flip.any():base[flip]=torch.flip(base[flip],(-1,));target[flip]=torch.flip(target[flip],(-1,))
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):prediction=refiner(base);loss,parts=_loss(prediction,target)
        loss.backward();torch.nn.utils.clip_grad_norm_(refiner.parameters(),1);optimizer.step()
        with torch.no_grad():
            for name,value in refiner.state_dict().items():ema[name].lerp_(value.detach(),1-training.ema_decay)
        if step==1 or step%250==0 or step==training.steps:
            row={"step":step,"loss":round(float(loss),6),"l1":round(float(parts[0]),6),"edge":round(float(parts[2]),6),"high":round(float(parts[3]),6)};history.append(row);print(json.dumps(row),flush=True)
    ema_cpu={name:value.detach().cpu() for name,value in ema.items()};recovery={"format":CHECKPOINT_FORMAT,"source_sha256":source_sha256(),"base_checkpoint_sha256":file_sha256(base_checkpoint),"corpus_sha256":corpus_sha,"model_config":config_dict(config),"training_config":config_dict(training),"ema_state":ema_cpu,"ema_sha256":_hash(ema_cpu),"history":history,"status":"trained_pending_evaluation"};torch.save(recovery,output/"trained_pending_evaluation.pt");refiner.load_state_dict(ema_cpu);refiner.to(device).eval();validation=targets[split:];base_validation=bases[split:];refined=[]
    with torch.inference_mode():
        for start in range(0,len(validation),8):refined.append(refiner(base_validation[start:start+8].to(device)).float().cpu())
    refined=torch.cat(refined);base_mae=float(F.l1_loss(base_validation,validation));refined_mae=float(F.l1_loss(refined,validation));base_mse=float(F.mse_loss(base_validation,validation));refined_mse=float(F.mse_loss(refined,validation));bdx,bdy=_edge(base_validation);rdx,rdy=_edge(refined);tdx,tdy=_edge(validation);base_edge=float(F.l1_loss(bdx,tdx)+F.l1_loss(bdy,tdy));refined_edge=float(F.l1_loss(rdx,tdx)+F.l1_loss(rdy,tdy));report={"format":"nullvector-world-frame-vae-pixel-refiner-training/1.0.0","source_sha256":source_sha256(),"base_checkpoint_sha256":file_sha256(base_checkpoint),"corpus_sha256":corpus_sha,"sources":list(sources),"refiner_parameters":sum(item.numel() for item in refiner.parameters()),"device":str(device),"steps":training.steps,"train_frames":split,"heldout_frames":len(validation),"heldout_base_mae":base_mae,"heldout_refined_mae":refined_mae,"heldout_base_mse":base_mse,"heldout_refined_mse":refined_mse,"heldout_base_psnr_db":-10*math.log10(max(base_mse,1e-12)),"heldout_refined_psnr_db":-10*math.log10(max(refined_mse,1e-12)),"heldout_base_edge_mae":base_edge,"heldout_refined_edge_mae":refined_edge,"mae_improvement":1-refined_mae/base_mae,"edge_improvement":1-refined_edge/base_edge,"history":history};payload={**recovery,"status":"evaluated","report":report};torch.save(payload,output/"checkpoint.pt");(output/"report.json").write_bytes(canonical(report));_contact(output/"heldout_contact_sheet.png",validation.permute(0,2,3,1).numpy(),base_validation.permute(0,2,3,1).numpy(),refined.permute(0,2,3,1).numpy());return report

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,required=True);parser.add_argument("--episodes",type=Path,nargs="+",required=True);parser.add_argument("--base",type=Path,required=True);parser.add_argument("--steps",type=int,default=5000);parser.add_argument("--batch-size",type=int,default=16);args=parser.parse_args();print(json.dumps(train(args.output,args.episodes,args.base,training=TrainingConfig(steps=args.steps,batch_size=args.batch_size)),indent=2))
