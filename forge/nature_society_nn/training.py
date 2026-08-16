from __future__ import annotations

import argparse,hashlib,json,random
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

from .contract import CHECKPOINT_FORMAT,ModelConfig,TrainingConfig,canonical,config_dict,source_sha256
from .corpus import build_corpus
from .model import SocietyStrategist


def _state_hash(state)->str:
    digest=hashlib.sha256()
    for name,value in sorted(state.items()):digest.update(name.encode()+b"\0"+value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def train(output:Path,*,model_config:ModelConfig=ModelConfig(),training:TrainingConfig=TrainingConfig())->dict:
    torch.manual_seed(training.seed);np.random.seed(training.seed&0xffffffff);random.seed(training.seed);torch.backends.cuda.matmul.allow_tf32=True;device=torch.device("cuda" if torch.cuda.is_available() else "cpu");corpus=build_corpus();split=int(len(corpus["features"])*.9);tensors={name:torch.from_numpy(value) for name,value in corpus.items() if isinstance(value,np.ndarray)};model=SocietyStrategist(model_config).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=training.learning_rate,weight_decay=2e-3,fused=device.type=="cuda");ema={name:value.detach().clone() for name,value in model.state_dict().items()};rng=np.random.default_rng(training.seed);history=[];model.train()
    for step in range(1,training.steps+1):
        index=torch.from_numpy(rng.integers(0,split,training.batch_size)).long();x=tensors["features"][index].to(device);targets=[tensors[name][index].to(device) for name in ("activity","labor","diplomacy","project")];optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):
            activity,labor_logits,diplomacy,project=model(x);labor=torch.softmax(labor_logits,-1);loss_activity=F.cross_entropy(activity,targets[0]);loss_labor=F.smooth_l1_loss(labor,targets[1]);loss_diplomacy=F.cross_entropy(diplomacy,targets[2]);loss_project=F.cross_entropy(project,targets[3]);loss=loss_activity+loss_diplomacy*.65+loss_project*.65+loss_labor*5
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
        with torch.no_grad():
            for name,value in model.state_dict().items():ema[name].lerp_(value.detach(),1-training.ema_decay)
        if step==1 or step%100==0 or step==training.steps:history.append({"step":step,"loss":round(float(loss),7),"activity":round(float(loss_activity),7),"labor":round(float(loss_labor),7),"diplomacy":round(float(loss_diplomacy),7),"project":round(float(loss_project),7)})
    live={name:value.detach().cpu() for name,value in model.state_dict().items()};ema_cpu={name:value.detach().cpu() for name,value in ema.items()};model.load_state_dict(ema_cpu);model.to(device).eval();correct=np.zeros(3,np.int64);total=0;labor_error=0.0
    with torch.inference_mode():
        for start in range(split,len(tensors["features"]),2048):
            x=tensors["features"][start:start+2048].to(device);targets=[tensors[name][start:start+2048].to(device) for name in ("activity","labor","diplomacy","project")];outputs=model(x);correct+=np.asarray([int((outputs[0].argmax(-1)==targets[0]).sum()),int((outputs[2].argmax(-1)==targets[2]).sum()),int((outputs[3].argmax(-1)==targets[3]).sum())]);labor_error+=float(torch.abs(torch.softmax(outputs[1],-1)-targets[1]).sum().cpu());total+=len(x)
    report={"format":"nullvector-neural-society-training-report/1.0.0","source_sha256":source_sha256(),"corpus_sha256":corpus["semantic_sha256"],"parameters":sum(p.numel() for p in model.parameters()),"device":str(device),"steps":training.steps,"heldout_activity_accuracy":float(correct[0]/total),"heldout_diplomacy_accuracy":float(correct[1]/total),"heldout_project_accuracy":float(correct[2]/total),"heldout_labor_mae":float(labor_error/(total*6)),"history":history};payload={"format":CHECKPOINT_FORMAT,"source_sha256":source_sha256(),"corpus_sha256":corpus["semantic_sha256"],"model_config":config_dict(model_config),"training_config":config_dict(training),"model_state":live,"ema_state":ema_cpu,"ema_sha256":_state_hash(ema_cpu),"report":report};runtime_payload={"format":CHECKPOINT_FORMAT,"source_sha256":source_sha256(),"corpus_sha256":corpus["semantic_sha256"],"model_config":config_dict(model_config),"training_config":config_dict(training),"ema_state":{name:value.to(torch.bfloat16) if value.is_floating_point() else value for name,value in ema_cpu.items()},"ema_sha256":_state_hash(ema_cpu),"report":report,"runtime_precision":"bfloat16"};output.mkdir(parents=True,exist_ok=False);torch.save(payload,output/"checkpoint.pt");torch.save(runtime_payload,output/"runtime.pt");(output/"report.json").write_bytes(canonical(report));return report


def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,required=True);parser.add_argument("--steps",type=int,default=1600);args=parser.parse_args();print(json.dumps(train(args.output,training=TrainingConfig(steps=args.steps)),indent=2))
