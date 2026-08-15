from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

import numpy as np
from PIL import Image,ImageDraw
import torch
from torch import Tensor
import torch.nn.functional as F

from ..organism_raster_vae_v3.calibration import _canonical,_font,_sha,_state_hash,source_sha256 as parent_source_sha256
from ..organism_raster_vae_v3.contract import RasterVAEV3Config
from ..organism_raster_vae_v3.model import StructuredRasterVAE
from ..organism_raster_vae_v3_appendage.calibration import _contact
from ..safety import require_disk_floor
from .dataset import GraphTokenCorpus
from .model import GraphTokenRasterVAE,loss


FORMAT="nullvector-graph-token-raster-vae-v4-calibration/1.0.0";CHECKPOINT_FORMAT="nullvector-graph-token-raster-vae-v4-checkpoint/1.0.0";SEED=0x4752415048544F4B
PARENT_MANIFEST=Path(__file__).resolve().parents[2]/"outputs/organism_raster_vae_v3/calibration_1200_alpha_scaffold/calibration_manifest.json"
SOURCE_FILES=("forge/organism_raster_vae_v4_graph/__init__.py","forge/organism_raster_vae_v4_graph/__main__.py","forge/organism_raster_vae_v4_graph/dataset.py","forge/organism_raster_vae_v4_graph/model.py","forge/organism_raster_vae_v4_graph/calibration.py","forge/organism_raster_vae_v4_graph/render_worker.py")


def source_manifest() -> dict[str,str]:
    root=Path(__file__).resolve().parents[2];return {relative:hashlib.sha256((root/relative).read_bytes()).hexdigest() for relative in SOURCE_FILES}
def source_sha256() -> str:return hashlib.sha256(_canonical({"files":source_manifest(),"parent":_sha(PARENT_MANIFEST),"parent_source":parent_source_sha256()})).hexdigest()
def _load_parent(device: torch.device):
    manifest=json.loads(PARENT_MANIFEST.read_text("utf-8"));descriptor=manifest["artifacts"]["checkpoint"];path=PARENT_MANIFEST.parent/descriptor["path"]
    if _sha(path)!=descriptor["sha256"] or manifest["source_sha256"]!=parent_source_sha256():raise ValueError("graph VAE parent drifted")
    payload=torch.load(path,map_location="cpu",weights_only=True);model=StructuredRasterVAE(RasterVAEV3Config(**payload["config"]));model.load_state_dict(payload["ema_state"]);return model.to(device).eval(),payload["ema_state"],manifest
def _warm(parent: dict[str,Tensor],device: torch.device):
    model=GraphTokenRasterVAE().to(device);state=model.state_dict();copied=0
    for name,value in parent.items():
        if name in state and state[name].shape==value.shape:state[name].copy_(value);copied+=1
    model.load_state_dict(state);return model,copied
def _batch(corpus,indices,device):
    rows=[corpus[index] for index in indices];return {key:torch.stack([row[key] for row in rows]).to(device) for key in rows[0]}


def _graph_contact(captures,baseline_capture):
    image=_contact(captures,baseline_capture);draw=ImageDraw.Draw(image);draw.rectangle((0,0,image.width,52),fill=(2,7,12));draw.text((14,8),"GRAPH-TOKEN VAE // HELD-OUT COMPARISON",font=_font(18),fill=(224,242,247));draw.text((14,31),"TARGET  /  PARENT VAE  /  GRAPH-OWNED APPENDAGE TOKENS",font=_font(10),fill=(78,219,239));cell=192
    for row in range(5):
        x=50+2*cell;y=59+row*(cell+36)+cell+3;draw.rectangle((x,y,x+150,y+16),fill=(2,7,12));draw.text((x,y+2),"GRAPH TOKEN VAE",font=_font(9),fill=(151,239,205))
    return image


@torch.inference_mode()
def _evaluate(model,corpus,indices,device,graph: bool):
    model.eval();sums={"alpha_iou":0.,"appendage_alpha_recall":0.,"appendage_neighborhood_precision":0.,"appendage_neighborhood_f1":0.,"rgba_mae":0.,"token_owner_accuracy":0.};count=0;owner_count=0;captures={}
    for start in range(0,len(indices),8):
        chosen=indices[start:start+8];batch=_batch(corpus,chosen,device)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            output=model(batch["living"],batch["family"],batch["traits"],batch["phase"],batch["tokens"],batch["token_mask"],stochastic=False) if graph else model(batch["living"],batch["family"],batch["traits"],batch["phase"],stochastic=False)
        prediction=output.rgba.float();alpha=batch["rgba"][:,3:]>.5;pa=prediction[:,3:]>.5;limb=batch["appendage_alpha"]>.5;tp=(pa&limb).flatten(1).sum(1).float();recall=tp/limb.flatten(1).sum(1).float().clamp_min(1);near=F.max_pool2d(limb.float(),5,1,2).bool();fp=(pa&near&~alpha).flatten(1).sum(1).float();precision=tp/(tp+fp).clamp_min(1);f1=2*precision*recall/(precision+recall).clamp_min(1e-8);inter=(alpha&pa).flatten(1).sum(1).float();union=(alpha|pa).flatten(1).sum(1).float().clamp_min(1);size=len(chosen);sums["alpha_iou"]+=float((inter/union).mean())*size;sums["appendage_alpha_recall"]+=float(recall.mean())*size;sums["appendage_neighborhood_precision"]+=float(precision.mean())*size;sums["appendage_neighborhood_f1"]+=float(f1.mean())*size;sums["rgba_mae"]+=float((prediction-batch["rgba"]).abs().mean())*size;count+=size
        if graph:
            owner=batch["token_owner"][:,::2,::2].reshape(size,-1);valid=owner>=0;correct=(output.token_attention.argmax(2)==owner)&valid;sums["token_owner_accuracy"]+=float(correct.sum());owner_count+=int(valid.sum())
        for local,index in enumerate(chosen):
            if corpus.rows[index][1]==8:captures[index]=(batch["rgba"][local].cpu(),prediction[local].cpu())
    result={key:round(value/count,9) for key,value in sums.items() if key!="token_owner_accuracy"};result["token_owner_accuracy"]=round(sums["token_owner_accuracy"]/max(owner_count,1),9) if graph else 0.;return result,captures


def calibrate(destination: Path,steps: int=600,batch_size: int=8)->Path:
    destination=destination.resolve()
    if destination.exists():raise FileExistsError(destination)
    require_disk_floor(destination.parent,floor_gb=100,planned_bytes=2*1024**3);device=torch.device("cuda");torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);torch.cuda.reset_peak_memory_stats(device);corpus=GraphTokenCorpus();parent,parent_state,parent_manifest=_load_parent(device);model,copied=_warm(parent_state,device);ema=copy.deepcopy(model).eval().requires_grad_(False);optimizer=torch.optim.AdamW(model.parameters(),lr=1.2e-4,weight_decay=1e-5,fused=True);order_gen=torch.Generator().manual_seed(SEED^0x4F5244);latent_gen=torch.Generator(device=device).manual_seed(SEED^0x4C4154);validation_ids={5,11,17,23,29};train=[i for i,(identity,_) in enumerate(corpus.rows) if identity not in validation_ids];validation=[i for i,(identity,_) in enumerate(corpus.rows) if identity in validation_ids];baseline,baseline_capture=_evaluate(parent,corpus,validation,device,False);order=torch.randperm(len(train),generator=order_gen).tolist();cursor=0;history=[];started=time.perf_counter();model.train()
    for step in range(steps):
        if cursor+batch_size>len(order):order=torch.randperm(len(train),generator=order_gen).tolist();cursor=0
        chosen=[train[order[cursor+i]] for i in range(batch_size)];cursor+=batch_size;batch=_batch(corpus,chosen,device);optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.bfloat16):output=model(batch["living"],batch["family"],batch["traits"],batch["phase"],batch["tokens"],batch["token_mask"],generator=latent_gen,stochastic=True);value,metrics=loss(output,batch,model.config,min(1,(step+1)/120))
        if not torch.isfinite(value):raise FloatingPointError("graph-token VAE became non-finite")
        value.backward();gradient=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1));optimizer.step()
        with torch.no_grad():torch._foreach_mul_(list(ema.parameters()),.996);torch._foreach_add_(list(ema.parameters()),list(model.parameters()),alpha=.004)
        if step==0 or (step+1)%20==0:history.append({"step":step+1,**{key:round(item,8) for key,item in metrics.items()},"gradient_norm":round(gradient,8),"gate12":round(float(torch.tanh(model.gate12).detach()),8),"gate24":round(float(torch.tanh(model.gate24).detach()),8)})
    seconds=time.perf_counter()-started;metrics,captures=_evaluate(ema,corpus,validation,device,True);staging=destination.parent/f".{destination.name}.tmp-{uuid.uuid4().hex}";staging.mkdir(parents=True)
    try:
        state={key:value.cpu() for key,value in model.state_dict().items()};ema_state={key:value.cpu() for key,value in ema.state_dict().items()};checkpoint={"format":CHECKPOINT_FORMAT,"source_sha256":source_sha256(),"parent_manifest_sha256":_sha(PARENT_MANIFEST),"config":model.config.to_dict(),"steps":steps,"model_state":state,"ema_state":ema_state,"model_state_sha256":_state_hash(state),"ema_state_sha256":_state_hash(ema_state),"history":history};path=staging/"graph_token_calibration.pt";torch.save(checkpoint,path)
        contact=staging/"graph_token_comparison.png";replay_report=staging/"visual_replay.json";subprocess.run([sys.executable,"-m","forge.organism_raster_vae_v4_graph.render_worker","--checkpoint",str(path),"--output",str(contact),"--report",str(replay_report)],cwd=Path(__file__).resolve().parents[2],check=True)
        replay=json.loads(replay_report.read_text("utf-8"));complete=bool(replay["complete"])
        if replay["metrics"]!=metrics or replay["baseline_metrics"]!=baseline or not complete:raise RuntimeError("graph token fresh-process visual replay differs")
        gates={"warm_start_parent_parameters_copied":copied>150,"all_families_held_out":len(captures)==5,"fresh_process_metric_replay_exact":True,"comparison_complete":complete,"token_owner_accuracy_above_random":metrics["token_owner_accuracy"]>.30,"appendage_f1_improved":metrics["appendage_neighborhood_f1"]>baseline["appendage_neighborhood_f1"],"appendage_recall_preserved":metrics["appendage_alpha_recall"]>=baseline["appendage_alpha_recall"]-.02,"alpha_iou_improved":metrics["alpha_iou"]>baseline["alpha_iou"],"production_promotion_allowed":False};manifest={"format":FORMAT,"status":"human_review_required","source_sha256":source_sha256(),"source_manifest":source_manifest(),"parent":{"file_sha256":_sha(PARENT_MANIFEST),"semantic_sha256":parent_manifest["manifest_sha256"]},"steps":steps,"batch_size":batch_size,"runtime":{"seconds":round(seconds,6),"device":torch.cuda.get_device_name(device),"parameters":sum(p.numel() for p in model.parameters()),"peak_allocated_bytes":torch.cuda.max_memory_allocated(device),"peak_reserved_bytes":torch.cuda.max_memory_reserved(device)},"baseline_metrics":baseline,"metrics":metrics,"loss_start":history[0]["loss"],"loss_end":history[-1]["loss"],"artifacts":{"checkpoint":{"path":path.name,"sha256":_sha(path),"bytes":path.stat().st_size},"contact":{"path":contact.name,"sha256":_sha(contact),"bytes":contact.stat().st_size},"visual_replay":{"path":replay_report.name,"sha256":_sha(replay_report),"bytes":replay_report.stat().st_size}},"gates":gates,"claim_boundary":{"graph_owned_appendage_tokens":True,"full_training_complete":False,"production_promotion_allowed":False}};manifest["manifest_sha256"]=hashlib.sha256(_canonical(manifest)).hexdigest();(staging/"graph_token_manifest.json").write_bytes(_canonical(manifest));staging.replace(destination)
    except BaseException:
        if staging.exists():shutil.rmtree(staging)
        raise
    return destination/"graph_token_manifest.json"
