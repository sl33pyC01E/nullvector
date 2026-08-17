from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import torch

from ..map_topology_neural_production.checkpoint import tensor_state_sha256
from ..safety import require_disk_floor
from .cache import build as build_cache, load as load_cache
from .contract import CHECKPOINT_FORMAT, DEFAULT_OUTPUT, FORMAT, Plan, canonical, sha256_file, source_sha256
from .model import NeuralCellRefiner, loss


CONTRACT_NAME="training_contract.json"; TELEMETRY_NAME="training_telemetry.json"


def checkpoint_name(step:int)->str:return f"refiner_{step:07d}.pt"


def _state_hash(state):return tensor_state_sha256({k:v.detach().cpu() for k,v in state.items()})


def _atomic(path:Path,value:bytes)->None:
    path.parent.mkdir(parents=True,exist_ok=True);descriptor,name=tempfile.mkstemp(prefix=f".{path.name}.tmp-",dir=path.parent);temporary=Path(name)
    try:
        with os.fdopen(descriptor,"wb") as handle:handle.write(value);handle.flush();os.fsync(handle.fileno())
        os.replace(temporary,path)
    finally:temporary.unlink(missing_ok=True)


def train(root:Path=DEFAULT_OUTPUT,plan:Plan=Plan())->dict[str,object]:
    root=Path(root).resolve();require_disk_floor(root.parent,floor_gb=100,planned_bytes=1024**3);root.mkdir(parents=True,exist_ok=True)
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8" or not torch.cuda.is_available():raise RuntimeError("V7 training requires deterministic CUDA")
    torch.use_deterministic_algorithms(True);build_cache(root)
    cache=load_cache(root);contract={"format":FORMAT,"source_sha256":source_sha256(),"plan":plan.to_dict(),"cache_sha256":sha256_file(root/"refiner_cache.pt"),"corpus_sha256":cache["corpus_sha256"],"model":{"width":64}}
    contract_path=root/CONTRACT_NAME
    if contract_path.exists() and json.loads(contract_path.read_bytes())!=contract:raise ValueError("V7 training contract drifted")
    if not contract_path.exists():_atomic(contract_path,canonical(contract))
    device=torch.device("cuda:0");torch.cuda.set_per_process_memory_fraction(.35,0);torch.set_num_threads(1);torch.manual_seed(plan.seed)
    model=NeuralCellRefiner(**contract["model"]).to(device);ema=copy.deepcopy(model).eval().requires_grad_(False);optimizer=torch.optim.AdamW(model.parameters(),lr=plan.learning_rate,weight_decay=1e-5,fused=True)
    generator=torch.Generator().manual_seed(plan.seed^0x4F52444552);history=[];start_step=0;latest=None
    for step in range(plan.segment_steps,plan.total_steps+1,plan.segment_steps):
        path=root/checkpoint_name(step)
        if path.exists():latest=path
    if latest is not None:
        payload=torch.load(latest,map_location="cpu",weights_only=True)
        if payload["format"]!=CHECKPOINT_FORMAT or payload["source_sha256"]!=source_sha256() or payload["contract"]!=contract:raise ValueError("V7 checkpoint drifted")
        model.load_state_dict(payload["model_state"],strict=True);ema.load_state_dict(payload["ema_state"],strict=True);optimizer.load_state_dict(payload["optimizer_state"]);generator.set_state(payload["generator_state"]);history=list(payload["history"]);start_step=int(payload["step"])
    train_indices=torch.tensor([i for i,v in enumerate(cache["identity"].tolist()) if int(v) not in {5,11,17,23,29}],dtype=torch.long)
    for end_step in range(start_step+plan.segment_steps,plan.total_steps+1,plan.segment_steps):
        torch.cuda.reset_peak_memory_stats(device);started=time.perf_counter();segment=[];model.train()
        for global_step in range(end_step-plan.segment_steps,end_step):
            chosen=train_indices[torch.randint(0,len(train_indices),(plan.batch_size,),generator=generator)]
            living=cache["living"][chosen].to(device);parent=cache["parent_rgba"][chosen].to(device);target=cache["target_rgba"][chosen].to(device);appendage=cache["appendage_alpha"][chosen].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.bfloat16):output=model(living,parent);value,metrics=loss(output,target,appendage)
            if not bool(torch.isfinite(value)):raise FloatingPointError("V7 training became non-finite")
            value.backward();gradient=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.0));optimizer.step()
            with torch.no_grad():torch._foreach_mul_(list(ema.parameters()),plan.ema_decay);torch._foreach_add_(list(ema.parameters()),list(model.parameters()),alpha=1-plan.ema_decay)
            if global_step==end_step-plan.segment_steps or (global_step+1)%25==0:
                row={"step":global_step+1,**{k:round(v,8) for k,v in metrics.items()},"gradient_norm":round(gradient,8)};history.append(row);segment.append(row)
        seconds=time.perf_counter()-started;model_state={k:v.detach().cpu() for k,v in model.state_dict().items()};ema_state={k:v.detach().cpu() for k,v in ema.state_dict().items()}
        payload={"format":CHECKPOINT_FORMAT,"source_sha256":source_sha256(),"contract":contract,"step":end_step,"model_state":model_state,"ema_state":ema_state,"optimizer_state":optimizer.state_dict(),"generator_state":generator.get_state(),"model_state_sha256":_state_hash(model_state),"ema_state_sha256":_state_hash(ema_state),"history":history,"runtime":{"seconds":round(seconds,6),"steps_per_second":round(plan.segment_steps/seconds,6),"peak_reserved_bytes":torch.cuda.max_memory_reserved(device),"device":torch.cuda.get_device_name(device)}}
        destination=root/checkpoint_name(end_step);temporary=root/f".{destination.name}.tmp-{os.getpid()}";torch.save(payload,temporary);os.replace(temporary,destination)
        telemetry={"format":FORMAT,"latest_step":end_step,"latest_checkpoint_sha256":sha256_file(destination),"latest_metrics":segment[-1],"runtime":payload["runtime"]};_atomic(root/TELEMETRY_NAME,canonical(telemetry))
    return {"passed":True,"step":plan.total_steps,"checkpoint":str(root/checkpoint_name(plan.total_steps)),"ema_state_sha256":payload["ema_state_sha256"]}


def load_final(root:Path=DEFAULT_OUTPUT):
    root=Path(root).resolve();contract=json.loads((root/CONTRACT_NAME).read_bytes());step=int(contract["plan"]["total_steps"]);payload=torch.load(root/checkpoint_name(step),map_location="cpu",weights_only=True)
    if payload["format"]!=CHECKPOINT_FORMAT or payload["source_sha256"]!=source_sha256() or payload["contract"]!=contract or payload["step"]!=step:raise ValueError("V7 final checkpoint drifted")
    model=NeuralCellRefiner(**contract["model"]);model.load_state_dict(payload["ema_state"],strict=True);model.eval();return model,payload,contract
