from __future__ import annotations

import copy,hashlib,json,os
from pathlib import Path
import tempfile,time

import torch

from ..map_topology_neural_production.checkpoint import tensor_state_sha256
from ..safety import require_disk_floor
from .cache import build as build_cache,load as load_cache
from .contract import CHECKPOINT_FORMAT,DEFAULT_OUTPUT,FORMAT,Plan,canonical,sha256_file,source_sha256
from .model import ContinuousCellVAE,loss


CONTRACT_NAME="training_contract.json";TELEMETRY_NAME="training_telemetry.json"
def checkpoint_name(step:int)->str:return f"cell_vae_{step:07d}.pt"
def _state_hash(state):return tensor_state_sha256({k:v.detach().cpu() for k,v in state.items()})
def _atomic(path:Path,value:bytes)->None:
    descriptor,name=tempfile.mkstemp(prefix=f".{path.name}.tmp-",dir=path.parent);temporary=Path(name)
    try:
        with os.fdopen(descriptor,"wb") as handle:handle.write(value);handle.flush();os.fsync(handle.fileno())
        os.replace(temporary,path)
    finally:temporary.unlink(missing_ok=True)


def train(root:Path=DEFAULT_OUTPUT,plan:Plan=Plan())->dict[str,object]:
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True);require_disk_floor(root.parent,floor_gb=100,planned_bytes=768*1024**2)
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG")!=":4096:8" or not torch.cuda.is_available():raise RuntimeError("cell VAE requires deterministic CUDA")
    torch.use_deterministic_algorithms(True);torch.set_num_threads(1);build_cache(root);cache=load_cache(root);contract={"format":FORMAT,"source_sha256":source_sha256(),"plan":plan.to_dict(),"cache_sha256":sha256_file(root/"cell_field_cache.pt"),"corpus_sha256":cache["corpus_sha256"],"model":{"width":96,"latent_dim":48}}
    contract_path=root/CONTRACT_NAME
    if contract_path.exists() and json.loads(contract_path.read_bytes())!=contract:raise ValueError("cell VAE contract drifted")
    if not contract_path.exists():_atomic(contract_path,canonical(contract))
    device=torch.device("cuda:0");torch.cuda.set_per_process_memory_fraction(.45,0);model=ContinuousCellVAE(**contract["model"]).to(device);ema=copy.deepcopy(model).eval().requires_grad_(False);optimizer=torch.optim.AdamW(model.parameters(),lr=plan.learning_rate,weight_decay=1e-5,fused=True);order=torch.Generator().manual_seed(plan.seed^0x4F52444552);latent=torch.Generator(device=device).manual_seed(plan.seed^0x4C4154454E54);history=[];start=0;payload=None
    for step in range(plan.segment_steps,plan.total_steps+1,plan.segment_steps):
        path=root/checkpoint_name(step)
        if path.exists():payload=torch.load(path,map_location="cpu",weights_only=True);start=step
    if payload is not None:
        if payload["format"]!=CHECKPOINT_FORMAT or payload["source_sha256"]!=source_sha256() or payload["contract"]!=contract:raise ValueError("cell VAE checkpoint drifted")
        model.load_state_dict(payload["model_state"]);ema.load_state_dict(payload["ema_state"]);optimizer.load_state_dict(payload["optimizer_state"]);order.set_state(payload["order_state"]);latent.set_state(payload["latent_state"]);history=list(payload["history"])
    reserved={4,5,10,11,16,17,22,23,28,29}
    train_indices=torch.tensor([i for i,v in enumerate(cache["identity"].tolist()) if int(v) not in reserved],dtype=torch.long)
    for end in range(start+plan.segment_steps,plan.total_steps+1,plan.segment_steps):
        torch.cuda.reset_peak_memory_stats(device);began=time.perf_counter();segment=[];model.train()
        for step in range(end-plan.segment_steps,end):
            chosen=train_indices[torch.randint(0,len(train_indices),(plan.batch_size,),generator=order)];features=cache["features"][chosen].to(device);mask=cache["mask"][chosen].to(device);target=cache["target_rgba"][chosen].to(device);optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.bfloat16):output=model(features,mask,generator=latent,stochastic=True);value,metrics=loss(output,target,mask,min(1,(step+1)/300))
            if not bool(torch.isfinite(value)):raise FloatingPointError("cell VAE became non-finite")
            value.backward();gradient=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1));optimizer.step()
            with torch.no_grad():torch._foreach_mul_(list(ema.parameters()),plan.ema_decay);torch._foreach_add_(list(ema.parameters()),list(model.parameters()),alpha=1-plan.ema_decay)
            if step==end-plan.segment_steps or (step+1)%25==0:row={"step":step+1,**{k:round(v,8) for k,v in metrics.items()},"gradient_norm":round(gradient,8)};history.append(row);segment.append(row)
        seconds=time.perf_counter()-began;model_state={k:v.detach().cpu() for k,v in model.state_dict().items()};ema_state={k:v.detach().cpu() for k,v in ema.state_dict().items()};payload={"format":CHECKPOINT_FORMAT,"source_sha256":source_sha256(),"contract":contract,"step":end,"model_state":model_state,"ema_state":ema_state,"optimizer_state":optimizer.state_dict(),"order_state":order.get_state(),"latent_state":latent.get_state().cpu(),"model_state_sha256":_state_hash(model_state),"ema_state_sha256":_state_hash(ema_state),"history":history,"runtime":{"seconds":round(seconds,6),"steps_per_second":round(plan.segment_steps/seconds,6),"peak_reserved_bytes":torch.cuda.max_memory_reserved(device),"device":torch.cuda.get_device_name(device)}};destination=root/checkpoint_name(end);temporary=root/f".{destination.name}.tmp-{os.getpid()}";torch.save(payload,temporary);os.replace(temporary,destination);_atomic(root/TELEMETRY_NAME,canonical({"format":FORMAT,"latest_step":end,"checkpoint_sha256":sha256_file(destination),"latest_metrics":segment[-1],"runtime":payload["runtime"]}))
    return {"passed":True,"step":plan.total_steps,"checkpoint":str(root/checkpoint_name(plan.total_steps)),"ema_state_sha256":payload["ema_state_sha256"]}


def load_final(root:Path=DEFAULT_OUTPUT):
    root=Path(root).resolve();contract=json.loads((root/CONTRACT_NAME).read_bytes());step=contract["plan"]["total_steps"];payload=torch.load(root/checkpoint_name(step),map_location="cpu",weights_only=True)
    if payload["format"]!=CHECKPOINT_FORMAT or payload["source_sha256"]!=source_sha256() or payload["contract"]!=contract:raise ValueError("cell VAE final checkpoint drifted")
    model=ContinuousCellVAE(**contract["model"]);model.load_state_dict(payload["ema_state"]);model.eval();return model,payload,contract
