from __future__ import annotations
import argparse,hashlib,json,random
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from .contract import CHECKPOINT_FORMAT,ModelConfig,TrainingConfig,canonical,config_dict,source_sha256
from .corpus import build_corpus
from .model import CounterfactualTransformer

def _hash(state):
    digest=hashlib.sha256()
    for name,value in sorted(state.items()):digest.update(name.encode()+b"\0"+value.cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()

def train(output:Path,*,config=ModelConfig(),training=TrainingConfig()):
    torch.manual_seed(training.seed);np.random.seed(training.seed&0xffffffff);random.seed(training.seed);torch.backends.cuda.matmul.allow_tf32=True;device=torch.device("cuda" if torch.cuda.is_available() else "cpu");data=build_corpus();groups=data["groups"];split_groups=int(groups*.9);split=split_groups*5;x=torch.from_numpy(data["sequence"]);a=torch.from_numpy(data["action"]);y=torch.from_numpy(data["target"]);score=torch.from_numpy(data["score"]);risk=torch.from_numpy(data["risk"]);model=CounterfactualTransformer(config).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=training.learning_rate,weight_decay=2e-3,fused=device.type=="cuda");ema={name:value.detach().clone() for name,value in model.state_dict().items()};rng=np.random.default_rng(training.seed);history=[]
    for step in range(1,training.steps+1):
        indices=torch.from_numpy(rng.integers(0,split,training.batch_size));bx=x[indices].to(device);ba=a[indices].to(device);by=y[indices].to(device);bs=score[indices].to(device);br=risk[indices].to(device);optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):pred,benefit,danger=model(bx,ba);state_loss=F.smooth_l1_loss(pred,by);score_loss=F.mse_loss(benefit,bs);risk_loss=F.mse_loss(danger,br);loss=state_loss*8+score_loss*2+risk_loss
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
        with torch.no_grad():
            for name,value in model.state_dict().items():ema[name].lerp_(value.detach(),1-training.ema_decay)
        if step==1 or step%100==0 or step==training.steps:history.append({"step":step,"loss":round(float(loss),6),"state":round(float(state_loss),6),"benefit":round(float(score_loss),6),"risk":round(float(risk_loss),6)})
    ema_cpu={name:value.detach().cpu() for name,value in ema.items()};model.load_state_dict(ema_cpu);model.eval();mae=score_mae=risk_mae=0;predicted=[];truth=[];total=0
    with torch.inference_mode():
        for start in range(split,len(x),256):pred,benefit,danger=model(x[start:start+256].to(device),a[start:start+256].to(device));count=len(pred);mae+=float((pred-y[start:start+count].to(device)).abs().sum());score_mae+=float((benefit-score[start:start+count].to(device)).abs().sum());risk_mae+=float((danger-risk[start:start+count].to(device)).abs().sum());predicted.extend(benefit.float().cpu().tolist());truth.extend(score[start:start+count].tolist());total+=count
    predicted=np.asarray(predicted).reshape(-1,5);truth=np.asarray(truth).reshape(-1,5);ranking=float((predicted.argmax(1)==truth.argmax(1)).mean());report={"format":"nullvector-neural-ecology-counterfactual-training/1.0.0","source_sha256":source_sha256(),"corpus_sha256":data["semantic_sha256"],"upstream_sha256":data["upstream_sha256"],"parameters":sum(parameter.numel() for parameter in model.parameters()),"device":str(device),"steps":training.steps,"heldout_state_mae":mae/(total*64),"heldout_benefit_mae":score_mae/total,"heldout_risk_mae":risk_mae/total,"heldout_top_action_accuracy":ranking,"history":history};payload={"format":CHECKPOINT_FORMAT,"source_sha256":source_sha256(),"corpus_sha256":data["semantic_sha256"],"model_config":config_dict(config),"training_config":config_dict(training),"ema_state":ema_cpu,"ema_sha256":_hash(ema_cpu),"report":report};output.mkdir(parents=True,exist_ok=False);torch.save(payload,output/"checkpoint.pt");(output/"report.json").write_bytes(canonical(report));return report

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,required=True);parser.add_argument("--steps",type=int,default=2800);parser.add_argument("--batch-size",type=int,default=96);args=parser.parse_args();print(json.dumps(train(args.output,training=TrainingConfig(steps=args.steps,batch_size=args.batch_size)),indent=2))
