from __future__ import annotations
import argparse,hashlib,json,random
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from .contract import CHECKPOINT_FORMAT,ModelConfig,TrainingConfig,canonical,config_dict,source_sha256
from .corpus import build_corpus
from .model import TimelineTransformer
def _hash(state):
    d=hashlib.sha256()
    for n,v in sorted(state.items()):d.update(n.encode()+b"\0"+v.cpu().contiguous().numpy().tobytes())
    return d.hexdigest()
def train(output:Path,*,config=ModelConfig(),training=TrainingConfig()):
    torch.manual_seed(training.seed);np.random.seed(training.seed&0xffffffff);random.seed(training.seed);torch.backends.cuda.matmul.allow_tf32=True;device=torch.device("cuda" if torch.cuda.is_available() else "cpu");data=build_corpus();x=torch.from_numpy(data["sequence"]);y=torch.from_numpy(data["target"]);event=torch.from_numpy(data["event"]);split=int(len(x)*.9);model=TimelineTransformer(config).to(device);opt=torch.optim.AdamW(model.parameters(),lr=training.learning_rate,weight_decay=2e-3,fused=device.type=="cuda");ema={n:v.detach().clone() for n,v in model.state_dict().items()};rng=np.random.default_rng(training.seed);history=[]
    for step in range(1,training.steps+1):
        idx=torch.from_numpy(rng.integers(0,split,training.batch_size));bx=x[idx].to(device);by=y[idx].to(device);be=event[idx].to(device);opt.zero_grad(set_to_none=True)
        with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):pred,logits,conf=model(bx);state=F.smooth_l1_loss(pred,by);classification=F.cross_entropy(logits,be);calibration=F.mse_loss(conf,(logits.argmax(-1)==be).float());loss=state*5+classification+calibration*.2
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
        with torch.no_grad():
            for n,v in model.state_dict().items():ema[n].lerp_(v.detach(),1-training.ema_decay)
        if step==1 or step%100==0 or step==training.steps:history.append({"step":step,"loss":round(float(loss),6),"state":round(float(state),6),"event":round(float(classification),6)})
    ema_cpu={n:v.detach().cpu() for n,v in ema.items()};model.load_state_dict(ema_cpu);model.eval();correct=total=0;mae=0
    with torch.inference_mode():
        for start in range(split,len(x),256):pred,logits,_=model(x[start:start+256].to(device));target=y[start:start+256].to(device);truth=event[start:start+256].to(device);mae+=float((pred-target).abs().sum());correct+=int((logits.argmax(-1)==truth).sum());total+=len(truth)
    report={"format":"nullvector-neural-world-timeline-training/1.0.0","source_sha256":source_sha256(),"corpus_sha256":data["semantic_sha256"],"parameters":sum(p.numel() for p in model.parameters()),"device":str(device),"steps":training.steps,"heldout_state_mae":mae/(total*64),"heldout_event_accuracy":correct/total,"history":history};payload={"format":CHECKPOINT_FORMAT,"source_sha256":source_sha256(),"corpus_sha256":data["semantic_sha256"],"model_config":config_dict(config),"training_config":config_dict(training),"ema_state":ema_cpu,"ema_sha256":_hash(ema_cpu),"report":report};output.mkdir(parents=True,exist_ok=False);torch.save(payload,output/"checkpoint.pt");(output/"report.json").write_bytes(canonical(report));return report
def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);p.add_argument("--steps",type=int,default=1400);p.add_argument("--batch-size",type=int,default=96);a=p.parse_args();print(json.dumps(train(a.output,training=TrainingConfig(steps=a.steps,batch_size=a.batch_size)),indent=2))
