from __future__ import annotations

import argparse,hashlib,json,random
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

from .contract import CHECKPOINT_FORMAT,ModelConfig,TrainingConfig,canonical,config_dict,source_sha256
from .corpus import build_corpus
from .model import ColonyCoordinator


def _state_hash(state)->str:
    digest=hashlib.sha256()
    for name,value in sorted(state.items()):digest.update(name.encode()+b"\0"+value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()

def train(output:Path,*,model_config:ModelConfig=ModelConfig(),training:TrainingConfig=TrainingConfig())->dict:
    torch.manual_seed(training.seed);np.random.seed(training.seed&0xffffffff);random.seed(training.seed);torch.backends.cuda.matmul.allow_tf32=True;device=torch.device("cuda" if torch.cuda.is_available() else "cpu");corpus=build_corpus();features=torch.from_numpy(corpus["features"]);roles=torch.from_numpy(corpus["roles"]);actions=torch.from_numpy(corpus["actions"]);mask=torch.from_numpy(corpus["mask"]);split=int(len(features)*.88);model=ColonyCoordinator(model_config).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=training.learning_rate,weight_decay=1e-3,fused=device.type=="cuda");ema={name:value.detach().clone() for name,value in model.state_dict().items()};rng=np.random.default_rng(training.seed);history=[]
    model.train()
    for step in range(1,training.steps+1):
        index=torch.from_numpy(rng.integers(0,split,training.batch_size)).long();x=features[index].to(device);m=mask[index].to(device);target_role=roles[index].to(device);target_action=actions[index].to(device);optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):
            role_logits,pred_action=model(x,m);role_loss=F.cross_entropy(role_logits.reshape(-1,6),target_role.reshape(-1),ignore_index=-100);valid=m.unsqueeze(-1);action_loss=((pred_action-target_action).square()*valid).sum()/valid.sum().clamp_min(1);loss=role_loss+action_loss*2
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
        with torch.no_grad():
            for name,value in model.state_dict().items():ema[name].lerp_(value.detach(),1-training.ema_decay)
        if step==1 or step%100==0 or step==training.steps:history.append({"step":step,"loss":round(float(loss),7),"role":round(float(role_loss),7),"action":round(float(action_loss),7)})
    live={name:value.detach().cpu() for name,value in model.state_dict().items()};ema_cpu={name:value.detach().cpu() for name,value in ema.items()};model.load_state_dict(ema_cpu);model.to(device).eval();correct=total=0;action_error=0.0
    with torch.inference_mode():
        for start in range(split,len(features),128):
            x=features[start:start+128].to(device);m=mask[start:start+128].to(device);target=roles[start:start+128].to(device);target_action=actions[start:start+128].to(device);logits,pred=model(x,m);valid=m;correct+=int(((logits.argmax(-1)==target)&valid).sum());total+=int(valid.sum());action_error+=float((torch.abs(pred-target_action)*valid.unsqueeze(-1)).sum().cpu())
    report={"format":"nullvector-neural-colony-training-report/1.0.0","source_sha256":source_sha256(),"corpus_sha256":corpus["semantic_sha256"],"parameters":sum(p.numel() for p in model.parameters()),"device":str(device),"steps":training.steps,"heldout_role_accuracy":correct/total,"heldout_action_mae":action_error/(total*3),"history":history}
    payload={"format":CHECKPOINT_FORMAT,"source_sha256":source_sha256(),"corpus_sha256":corpus["semantic_sha256"],"model_config":config_dict(model_config),"training_config":config_dict(training),"model_state":live,"ema_state":ema_cpu,"ema_sha256":_state_hash(ema_cpu),"report":report};output.mkdir(parents=True,exist_ok=False);torch.save(payload,output/"checkpoint.pt");(output/"report.json").write_bytes(canonical(report));return report

def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,required=True);parser.add_argument("--steps",type=int,default=1200);args=parser.parse_args();print(json.dumps(train(args.output,training=TrainingConfig(steps=args.steps)),indent=2))
if __name__=="__main__":main()
