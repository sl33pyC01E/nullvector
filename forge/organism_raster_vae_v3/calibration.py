from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
import uuid

import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
from torch import Tensor

from ..safety import require_disk_floor
from .contract import CHECKPOINT_FORMAT,FORMAT,RasterVAEV3Config
from .dataset import MorphologyMotionCorpus
from .model import StructuredRasterVAE,loss


SOURCE_FILES=("forge/organism_raster_vae_v3/__init__.py","forge/organism_raster_vae_v3/__main__.py","forge/organism_raster_vae_v3/contract.py","forge/organism_raster_vae_v3/dataset.py","forge/organism_raster_vae_v3/model.py","forge/organism_raster_vae_v3/calibration.py")
SEED=0x5641453352415354


def _canonical(payload: object) -> bytes: return (json.dumps(payload,sort_keys=True,indent=2,allow_nan=False)+"\n").encode("utf-8")
def _sha(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(4*1024*1024),b""):digest.update(block)
    return digest.hexdigest()
def source_manifest() -> dict[str,str]:
    root=Path(__file__).resolve().parents[2]; return {relative:hashlib.sha256((root/relative).read_bytes()).hexdigest() for relative in SOURCE_FILES}
def source_sha256() -> str: return hashlib.sha256(_canonical(source_manifest())).hexdigest()
def _state_hash(state: dict[str,Tensor]) -> str:
    digest=hashlib.sha256(b"nullvector-vae-v3-state\0")
    for name in sorted(state):
        value=state[name].detach().cpu().contiguous(); digest.update(name.encode()+b"\0"+str(value.dtype).encode()+b"\0"+np.asarray(value.shape,dtype="<i8").tobytes()+memoryview(value.numpy()))
    return digest.hexdigest()
def _font(size: int):
    path=Path("C:/Windows/Fonts/consola.ttf"); return ImageFont.truetype(str(path),size) if path.is_file() else ImageFont.load_default()


def _batch(corpus: MorphologyMotionCorpus,indices: list[int],device: torch.device) -> dict[str,Tensor]:
    rows=[corpus[index] for index in indices]; return {key:torch.stack([row[key] for row in rows]).to(device) for key in rows[0]}


@torch.inference_mode()
def _evaluate(model: StructuredRasterVAE,corpus: MorphologyMotionCorpus,indices: list[int],device: torch.device,batch_size: int=4) -> tuple[dict[str,float],dict[int,tuple[dict[str,Tensor],Tensor]]]:
    model.eval(); sums={"rgba_mae":0.0,"foreground_rgb_mae":0.0,"alpha_iou":0.0,"tissue_accuracy_visible":0.0,"edge_mae":0.0}; count=0; captures={}
    for start in range(0,len(indices),batch_size):
        chosen=indices[start:start+batch_size]; batch=_batch(corpus,chosen,device)
        with torch.autocast("cuda",dtype=torch.bfloat16): output=model(batch["living"],batch["family"],batch["traits"],batch["phase"],stochastic=False)
        predicted=output.rgba.float(); target=batch["rgba"].float(); alpha=target[:,3:]>.5; pa=predicted[:,3:]>.5
        intersection=(alpha&pa).flatten(1).sum(1).float(); union=(alpha|pa).flatten(1).sum(1).float().clamp_min(1); visible=alpha.float(); visible_count=visible.sum().clamp_min(1)
        edge=sum((a-b).abs().mean() for a,b in ((predicted[:,:,:,1:]-predicted[:,:,:,:-1],target[:,:,:,1:]-target[:,:,:,:-1]),(predicted[:,:,1:,:]-predicted[:,:,:-1,:],target[:,:,1:,:]-target[:,:,:-1,:])))
        size=len(chosen); sums["rgba_mae"]+=float((predicted-target).abs().mean())*size; sums["foreground_rgb_mae"]+=float(((predicted[:,:3]-target[:,:3]).abs()*visible).sum()/(visible_count*3))*size; sums["alpha_iou"]+=float((intersection/union).mean())*size; sums["tissue_accuracy_visible"]+=float(((output.tissue_logits.argmax(1)==batch["tissue"])*batch["occupancy"].bool()).sum()/batch["occupancy"].sum().clamp_min(1))*size; sums["edge_mae"]+=float(edge)*size; count+=size
        for local,index in enumerate(chosen):
            # One stable mid-cycle representative for each held-out identity.
            if corpus.rows[index][1]==8: captures[index]=({key:value[local].detach().cpu() for key,value in batch.items()},predicted[local].detach().cpu())
    return {key:round(value/count,9) for key,value in sums.items()},captures


def _image(tensor: Tensor) -> Image.Image:
    value=tensor.float().clamp(0,1); composite=torch.cat((value[:3]*value[3:],torch.ones_like(value[3:])),0)
    array=(composite.permute(1,2,0).numpy()*255+.5).astype(np.uint8); return Image.fromarray(array,"RGBA").convert("RGB")


def _contact(captures: dict[int,tuple[dict[str,Tensor],Tensor]]) -> Image.Image:
    scale=2; cell=96*scale; width=32+2*(cell+18); height=62+5*(cell+42)
    canvas=Image.new("RGB",(width,height),(3,8,14)); draw=ImageDraw.Draw(canvas); draw.text((16,10),"VAE V3 CALIBRATION // TARGET vs NEURAL RECONSTRUCTION",font=_font(19),fill=(222,242,248)); draw.text((16,34),"96px continuous cellular raster · 114M parameters · morphology-v2 corpus",font=_font(11),fill=(76,216,239))
    for row,index in enumerate(sorted(captures)):
        batch,predicted=captures[index]; y=62+row*(cell+42); family=int(batch["family"]); labels=("HUMANOID","ANIMALIAN","PLANTLIKE","ANOMALY","MACHINE")
        target=_image(batch["rgba"]).resize((cell,cell),Image.Resampling.NEAREST); recon=_image(predicted).resize((cell,cell),Image.Resampling.NEAREST)
        canvas.paste(target,(16,y)); canvas.paste(recon,(34+cell,y)); draw.text((16,y+cell+5),f"{labels[family]} // TARGET",font=_font(10),fill=(160,185,195)); draw.text((34+cell,y+cell+5),"NEURAL MEAN DECODE",font=_font(10),fill=(160,238,214))
    return canvas


def calibrate(destination: Path,*,steps: int=120,batch_size: int=4) -> Path:
    destination=destination.resolve()
    if destination.exists(): raise FileExistsError(destination)
    if not 40<=steps<=1600 or not 1<=batch_size<=12: raise ValueError("VAE v3 calibration bounds drifted")
    require_disk_floor(destination.parent,floor_gb=100,planned_bytes=2*1024**3)
    if not torch.cuda.is_available(): raise RuntimeError("VAE v3 calibration requires CUDA")
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED); torch.backends.cuda.matmul.allow_tf32=True
    device=torch.device("cuda"); torch.cuda.reset_peak_memory_stats(device); corpus=MorphologyMotionCorpus(); config=RasterVAEV3Config(); model=StructuredRasterVAE(config).to(device); ema=copy.deepcopy(model).eval().requires_grad_(False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-5,fused=True); order_gen=torch.Generator().manual_seed(SEED^0x4F52444552); latent_gen=torch.Generator(device=device).manual_seed(SEED^0x4C4154454E54)
    validation_identities={5,11,17,23,29}; train=[index for index,(identity,_) in enumerate(corpus.rows) if identity not in validation_identities]; validation=[index for index,(identity,_) in enumerate(corpus.rows) if identity in validation_identities]
    order=torch.randperm(len(train),generator=order_gen).tolist(); cursor=0; history=[]; started=time.perf_counter(); model.train()
    for step in range(steps):
        if cursor+batch_size>len(order): order=torch.randperm(len(train),generator=order_gen).tolist(); cursor=0
        chosen=[train[order[cursor+offset]] for offset in range(batch_size)]; cursor+=batch_size; batch=_batch(corpus,chosen,device); optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            output=model(batch["living"],batch["family"],batch["traits"],batch["phase"],generator=latent_gen,stochastic=True); value,metrics=loss(output,batch,config,min(1.0,(step+1)/80))
        if not torch.isfinite(value): raise FloatingPointError("VAE v3 calibration loss became non-finite")
        value.backward(); gradient=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)); optimizer.step()
        with torch.no_grad():
            torch._foreach_mul_(list(ema.parameters()),.995); torch._foreach_add_(list(ema.parameters()),list(model.parameters()),alpha=.005)
            for left,right in zip(ema.buffers(),model.buffers(),strict=True): left.copy_(right)
        if step==0 or (step+1)%10==0: history.append({"step":step+1,**{key:round(item,8) for key,item in metrics.items()},"gradient_norm":round(gradient,8)})
    seconds=time.perf_counter()-started; metrics,captures=_evaluate(ema,corpus,validation,device); staging=destination.parent/f".{destination.name}.tmp-{uuid.uuid4().hex}"; staging.mkdir(parents=True)
    try:
        contact=staging/"reconstruction_contact.png"; _contact(captures).save(contact,compress_level=7)
        state={key:value.detach().cpu() for key,value in model.state_dict().items()}; ema_state={key:value.detach().cpu() for key,value in ema.state_dict().items()}; checkpoint={"format":CHECKPOINT_FORMAT,"source_sha256":source_sha256(),"config":config.to_dict(),"steps":steps,"batch_size":batch_size,"seed":SEED,"corpus_sha256":corpus.semantic_sha256,"model_state":state,"ema_state":ema_state,"model_state_sha256":_state_hash(state),"ema_state_sha256":_state_hash(ema_state),"history":history}; checkpoint_path=staging/"calibration.pt"; torch.save(checkpoint,checkpoint_path)
        manifest={"format":FORMAT,"status":"human_review_required","source_sha256":source_sha256(),"source_manifest":source_manifest(),"config":config.to_dict(),"steps":steps,"batch_size":batch_size,"seed":SEED,"corpus":{"semantic_sha256":corpus.semantic_sha256,"samples":len(corpus),"train":len(train),"validation":len(validation),"identities":30,"phases":16},"runtime":{"device":torch.cuda.get_device_name(device),"precision":"bf16-autocast-float32-loss","seconds":round(seconds,6),"steps_per_second":round(steps/seconds,6),"peak_allocated_bytes":torch.cuda.max_memory_allocated(device),"peak_reserved_bytes":torch.cuda.max_memory_reserved(device),"parameters":sum(parameter.numel() for parameter in model.parameters())},"metrics":metrics,"loss_start":history[0]["loss"],"loss_end":history[-1]["loss"],"artifacts":{"checkpoint":{"path":checkpoint_path.name,"sha256":_sha(checkpoint_path),"bytes":checkpoint_path.stat().st_size},"contact":{"path":contact.name,"sha256":_sha(contact),"bytes":contact.stat().st_size}},"gates":{"finite_training":all(math.isfinite(row["loss"]) for row in history),"loss_improved":history[-1]["loss"]<history[0]["loss"],"all_families_held_out":len(captures)==5,"high_resolution_96px":True,"production_promotion_allowed":False},"claim_boundary":{"architecture_and_data_path_calibrated":True,"full_training_complete":False,"human_morphology_approval_required":True,"old_v2_frozen_rasterizer_replaced":False}}
        manifest["manifest_sha256"]=hashlib.sha256(_canonical(manifest)).hexdigest(); manifest_path=staging/"calibration_manifest.json"; manifest_path.write_bytes(_canonical(manifest)); staging.replace(destination)
    except BaseException:
        if staging.exists(): shutil.rmtree(staging)
        raise
    return destination/"calibration_manifest.json"
