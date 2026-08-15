from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil
import time
import uuid

import numpy as np
from PIL import Image,ImageDraw
import torch
from torch import Tensor
import torch.nn.functional as F

from ..organism_raster_vae_v3.calibration import _canonical,_font,_image,_sha,_state_hash,source_sha256 as parent_source_sha256
from ..organism_raster_vae_v3.contract import RasterVAEV3Config
from ..organism_raster_vae_v3.model import StructuredRasterVAE
from ..safety import require_disk_floor
from .dataset import APPENDAGE_CLASSES,AppendageMotionCorpus
from .model import AppendageRasterVAE,loss


FORMAT="nullvector-appendage-aware-raster-vae-v3-continuation/1.0.0"; CHECKPOINT_FORMAT="nullvector-appendage-aware-raster-vae-v3-checkpoint/1.0.0"; SEED=0x415050454E444147
PARENT_MANIFEST=Path(__file__).resolve().parents[2]/"outputs/organism_raster_vae_v3/calibration_1200_alpha_scaffold/calibration_manifest.json"
SOURCE_FILES=("forge/organism_raster_vae_v3_appendage/__init__.py","forge/organism_raster_vae_v3_appendage/__main__.py","forge/organism_raster_vae_v3_appendage/dataset.py","forge/organism_raster_vae_v3_appendage/model.py","forge/organism_raster_vae_v3_appendage/calibration.py")


def source_manifest() -> dict[str,str]:
    root=Path(__file__).resolve().parents[2]; return {relative:hashlib.sha256((root/relative).read_bytes()).hexdigest() for relative in SOURCE_FILES}
def source_sha256() -> str: return hashlib.sha256(_canonical({"files":source_manifest(),"parent_manifest_sha256":_sha(PARENT_MANIFEST),"parent_source_sha256":parent_source_sha256()})).hexdigest()


def _load_parent(device: torch.device) -> tuple[StructuredRasterVAE,dict[str,Tensor],dict[str,object]]:
    manifest=json.loads(PARENT_MANIFEST.read_text("utf-8")); descriptor=manifest["artifacts"]["checkpoint"]; path=PARENT_MANIFEST.parent/descriptor["path"]
    if _sha(path)!=descriptor["sha256"] or manifest["source_sha256"]!=parent_source_sha256(): raise ValueError("appendage continuation parent drifted")
    payload=torch.load(path,map_location="cpu",weights_only=True); model=StructuredRasterVAE(RasterVAEV3Config(**payload["config"])); model.load_state_dict(payload["ema_state"],strict=True); return model.to(device).eval(),payload["ema_state"],manifest


def _warm_model(parent_state: dict[str,Tensor],device: torch.device) -> AppendageRasterVAE:
    model=AppendageRasterVAE().to(device); state=model.state_dict(); copied=[]
    for name,value in parent_state.items():
        if name in state and state[name].shape==value.shape: state[name].copy_(value); copied.append(name)
    state["stem.weight"][:,:42].copy_(parent_state["stem.weight"]); state["stem.weight"][:,42:].zero_(); state["stem.bias"].copy_(parent_state["stem.bias"])
    state["render.2.weight"][:4].copy_(parent_state["render.2.weight"]); state["render.2.bias"][:4].copy_(parent_state["render.2.bias"]); state["render.2.weight"][4:].zero_(); state["render.2.bias"][4:].zero_(); model.load_state_dict(state,strict=True)
    if len(copied)<len(parent_state)-4: raise ValueError("appendage continuation warm-start coverage drifted")
    return model


def _batch(corpus: AppendageMotionCorpus,indices: list[int],device: torch.device) -> dict[str,Tensor]:
    rows=[corpus[index] for index in indices]; return {key:torch.stack([row[key] for row in rows]).to(device) for key in rows[0]}


@torch.inference_mode()
def _evaluate(model,corpus: AppendageMotionCorpus,indices: list[int],device: torch.device,batch_size: int=8):
    model.eval(); sums={"alpha_iou":0.0,"appendage_alpha_recall":0.0,"appendage_neighborhood_f1":0.0,"appendage_neighborhood_precision":0.0,"rgba_mae":0.0}; count=0; captures={}
    for start in range(0,len(indices),batch_size):
        chosen=indices[start:start+batch_size]; batch=_batch(corpus,chosen,device)
        with torch.autocast("cuda",dtype=torch.bfloat16): output=model(batch["living"][:,:42] if isinstance(model,StructuredRasterVAE) and not isinstance(model,AppendageRasterVAE) else batch["living"],batch["family"],batch["traits"],batch["phase"],stochastic=False)
        predicted=output.rgba.float(); alpha=batch["rgba"][:,3:]>.5; pa=predicted[:,3:]>.5; limb=batch["appendage_alpha"]>.5; intersection=(alpha&pa).flatten(1).sum(1).float(); union=(alpha|pa).flatten(1).sum(1).float().clamp_min(1); true_positive=(pa&limb).flatten(1).sum(1).float(); limb_recall=true_positive/limb.flatten(1).sum(1).float().clamp_min(1); neighborhood=F.max_pool2d(limb.float(),5,stride=1,padding=2).bool(); false_positive=(pa&neighborhood&~alpha).flatten(1).sum(1).float(); limb_precision=true_positive/(true_positive+false_positive).clamp_min(1); limb_f1=2*limb_precision*limb_recall/(limb_precision+limb_recall).clamp_min(1e-8); size=len(chosen); sums["alpha_iou"]+=float((intersection/union).mean())*size; sums["appendage_alpha_recall"]+=float(limb_recall.mean())*size; sums["appendage_neighborhood_precision"]+=float(limb_precision.mean())*size; sums["appendage_neighborhood_f1"]+=float(limb_f1.mean())*size; sums["rgba_mae"]+=float((predicted-batch["rgba"]).abs().mean())*size; count+=size
        for local,index in enumerate(chosen):
            if corpus.rows[index][1]==8: captures[index]=(batch["rgba"][local].cpu(),predicted[local].cpu())
    return {key:round(value/count,9) for key,value in sums.items()},captures


def _contact(targets: dict[int,tuple[Tensor,Tensor]],baseline: dict[int,tuple[Tensor,Tensor]]) -> Image.Image:
    cell=192;width=48+3*(cell+18);height=59+5*(cell+36);canvas=Image.new("RGB",(width,height),(2,7,12));draw=ImageDraw.Draw(canvas);draw.text((14,8),"APPENDAGE-AWARE VAE // HELD-OUT COMPARISON",font=_font(18),fill=(224,242,247));draw.text((14,31),"TARGET  /  PARENT VAE  /  APPENDAGE-SUPERVISED CONTINUATION",font=_font(10),fill=(78,219,239))
    names=("HUMANOID","ANIMALIAN","PLANTLIKE","ANOMALY","MACHINE")
    for row,index in enumerate(sorted(targets)):
        target,new=targets[index]; old=baseline[index][1];y=59+row*(cell+36)
        for col,value in enumerate((target,old,new)):canvas.paste(_image(value).resize((cell,cell),Image.Resampling.NEAREST),(14+col*(cell+18),y))
        draw.text((14,y+cell+5),names[row],font=_font(9),fill=(165,191,201));draw.text((32+cell,y+cell+5),"PARENT",font=_font(9),fill=(164,146,214));draw.text((50+2*cell,y+cell+5),"APPENDAGE VAE",font=_font(9),fill=(151,239,205))
    return canvas


def calibrate(destination: Path,steps: int=600,batch_size: int=8) -> Path:
    destination=destination.resolve()
    if destination.exists():raise FileExistsError(destination)
    require_disk_floor(destination.parent,floor_gb=100,planned_bytes=2*1024**3);device=torch.device("cuda");torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);torch.cuda.reset_peak_memory_stats(device);corpus=AppendageMotionCorpus();parent,parent_state,parent_manifest=_load_parent(device);model=_warm_model(parent_state,device);ema=copy.deepcopy(model).eval().requires_grad_(False);optimizer=torch.optim.AdamW(model.parameters(),lr=1.2e-4,weight_decay=1e-5,fused=True);order_gen=torch.Generator().manual_seed(SEED^0x4F5244);latent_gen=torch.Generator(device=device).manual_seed(SEED^0x4C4154);validation_ids={5,11,17,23,29};train=[i for i,(identity,_) in enumerate(corpus.rows) if identity not in validation_ids];validation=[i for i,(identity,_) in enumerate(corpus.rows) if identity in validation_ids];baseline,baseline_captures=_evaluate(parent,corpus,validation,device);order=torch.randperm(len(train),generator=order_gen).tolist();cursor=0;history=[];started=time.perf_counter();model.train()
    for step in range(steps):
        if cursor+batch_size>len(order):order=torch.randperm(len(train),generator=order_gen).tolist();cursor=0
        chosen=[train[order[cursor+i]] for i in range(batch_size)];cursor+=batch_size;batch=_batch(corpus,chosen,device);optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.bfloat16):output=model(batch["living"],batch["family"],batch["traits"],batch["phase"],generator=latent_gen,stochastic=True);value,metrics=loss(output,batch,model.config,min(1,(step+1)/120))
        if not torch.isfinite(value):raise FloatingPointError("appendage continuation became non-finite")
        value.backward();gradient=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1));optimizer.step()
        with torch.no_grad():torch._foreach_mul_(list(ema.parameters()),.996);torch._foreach_add_(list(ema.parameters()),list(model.parameters()),alpha=.004)
        if step==0 or (step+1)%20==0:history.append({"step":step+1,**{key:round(item,8) for key,item in metrics.items()},"gradient_norm":round(gradient,8)})
    seconds=time.perf_counter()-started;metrics,captures=_evaluate(ema,corpus,validation,device);staging=destination.parent/f".{destination.name}.tmp-{uuid.uuid4().hex}";staging.mkdir(parents=True)
    try:
        contact=staging/"appendage_comparison.png";contact_image=_contact(captures,baseline_captures);contact_image.save(contact,compress_level=7);decoded=np.asarray(Image.open(contact).convert("RGB"));cell=192
        contact_complete=all(int(np.count_nonzero(decoded[59+row*(cell+36):59+row*(cell+36)+cell,14+col*(cell+18):14+col*(cell+18)+cell].max(2)>24))>300 for row in range(5) for col in range(3))
        if not contact_complete:raise RuntimeError("appendage comparison artifact is incomplete")
        state={key:value.cpu() for key,value in model.state_dict().items()};ema_state={key:value.cpu() for key,value in ema.state_dict().items()};checkpoint={"format":CHECKPOINT_FORMAT,"source_sha256":source_sha256(),"parent_manifest_sha256":_sha(PARENT_MANIFEST),"config":model.config.to_dict(),"steps":steps,"model_state":state,"ema_state":ema_state,"model_state_sha256":_state_hash(state),"ema_state_sha256":_state_hash(ema_state),"history":history};path=staging/"appendage_calibration.pt";torch.save(checkpoint,path);gates={"warm_start_exact":True,"all_families_held_out":len(captures)==5,"comparison_artifact_complete":contact_complete,"appendage_neighborhood_f1_improved":metrics["appendage_neighborhood_f1"]>baseline["appendage_neighborhood_f1"],"appendage_recall_preserved":metrics["appendage_alpha_recall"]>=baseline["appendage_alpha_recall"]-.02,"alpha_iou_improved":metrics["alpha_iou"]>baseline["alpha_iou"],"production_promotion_allowed":False};manifest={"format":FORMAT,"status":"human_review_required","source_sha256":source_sha256(),"source_manifest":source_manifest(),"parent":{"manifest_sha256":_sha(PARENT_MANIFEST),"semantic_sha256":parent_manifest["manifest_sha256"]},"steps":steps,"batch_size":batch_size,"runtime":{"seconds":round(seconds,6),"device":torch.cuda.get_device_name(device),"parameters":sum(p.numel() for p in model.parameters()),"peak_allocated_bytes":torch.cuda.max_memory_allocated(device),"peak_reserved_bytes":torch.cuda.max_memory_reserved(device)},"baseline_metrics":baseline,"metrics":metrics,"loss_start":history[0]["loss"],"loss_end":history[-1]["loss"],"artifacts":{"checkpoint":{"path":path.name,"sha256":_sha(path),"bytes":path.stat().st_size},"contact":{"path":contact.name,"sha256":_sha(contact),"bytes":contact.stat().st_size}},"gates":gates,"claim_boundary":{"appendage_aware_continuation":True,"two_prior_attempts_preserved_as_failed_evidence":True,"full_training_complete":False,"production_promotion_allowed":False}};manifest["manifest_sha256"]=hashlib.sha256(_canonical(manifest)).hexdigest();(staging/"appendage_manifest.json").write_bytes(_canonical(manifest));staging.replace(destination)
    except BaseException:
        if staging.exists():shutil.rmtree(staging)
        raise
    return destination/"appendage_manifest.json"
