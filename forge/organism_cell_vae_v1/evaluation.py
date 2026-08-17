from __future__ import annotations

import hashlib,io,json,os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
from PIL import Image,ImageDraw
import torch

from .cache import load as load_cache
from .contract import DEFAULT_OUTPUT,EVALUATION_FORMAT,canonical,sha256_file,source_sha256
from .training import load_final


MANIFEST_NAME="evaluation_manifest.json";SHEET_NAME="continuous_cell_vae_heldout.png"
def _atomic(path:Path,value:bytes)->None:
    descriptor,name=tempfile.mkstemp(prefix=f".{path.name}.tmp-",dir=path.parent);temporary=Path(name)
    try:
        with os.fdopen(descriptor,"wb") as handle:handle.write(value);handle.flush();os.fsync(handle.fileno())
        os.replace(temporary,path)
    finally:temporary.unlink(missing_ok=True)
def _image(value:torch.Tensor)->Image.Image:return Image.fromarray((value.detach().cpu().permute(1,2,0).numpy().clip(0,1)*255+.5).astype(np.uint8),"RGBA")


@torch.inference_mode()
def evaluate(root:Path=DEFAULT_OUTPUT,*,device_name:str="cuda")->dict[str,Any]:
    root=Path(root).resolve()
    if (root/MANIFEST_NAME).exists():raise FileExistsError("cell VAE evaluation is immutable")
    cache=load_cache(root);model,checkpoint,contract=load_final(root);device=torch.device(device_name);model.to(device)
    calibration_indices=[i for i,v in enumerate(cache["identity"].tolist()) if int(v) in {4,10,16,22,28}]
    indices=[i for i,v in enumerate(cache["identity"].tolist()) if int(v) in {5,11,17,23,29}]
    threshold_scores={round(value/40,3):[0.,0.] for value in range(12,29)}
    for start in range(0,len(calibration_indices),4):
        chosen=torch.tensor(calibration_indices[start:start+4]);prediction=model(cache["features"][chosen].to(device),cache["mask"][chosen].to(device),stochastic=False).rgba[:,3:];alpha=cache["target_rgba"][chosen,3:].to(device)>.5
        for threshold,values in threshold_scores.items():
            visible=prediction>threshold;values[0]+=float((visible&alpha).flatten(1).sum(1).sum());values[1]+=float((visible|alpha).flatten(1).sum(1).sum())
    threshold=max(threshold_scores,key=lambda value:threshold_scores[value][0]/max(threshold_scores[value][1],1));calibration_iou=threshold_scores[threshold][0]/threshold_scores[threshold][1]
    sums={"alpha_iou":0.,"rgba_mae":0.,"foreground_rgb_mae":0.};count=0;captures=[]
    for start in range(0,len(indices),4):
        chosen=torch.tensor(indices[start:start+4]);features=cache["features"][chosen].to(device);mask=cache["mask"][chosen].to(device);target=cache["target_rgba"][chosen].to(device).float();prediction=model(features,mask,stochastic=False).rgba.float();alpha=target[:,3:]>.5;pa=prediction[:,3:]>threshold;inter=(alpha&pa).flatten(1).sum(1).float();union=(alpha|pa).flatten(1).sum(1).float().clamp_min(1);fg=alpha.float();sums["alpha_iou"]+=float((inter/union).sum());sums["rgba_mae"]+=float((prediction-target).abs().flatten(1).mean(1).sum());sums["foreground_rgb_mae"]+=float((((prediction[:,:3]-target[:,:3]).abs()*fg).sum((1,2,3))/(fg.sum((1,2,3))*3).clamp_min(1)).sum());count+=len(chosen)
        for local,index in enumerate(chosen.tolist()):
            if int(cache["phase_index"][index])==8:captures.append((int(cache["identity"][index]),target[local].cpu(),prediction[local].cpu()))
    metrics={k:round(v/count,8) for k,v in sums.items()};metrics["calibrated_alpha_threshold"]=threshold;metrics["calibration_alpha_iou"]=round(calibration_iou,8);tile=192;canvas=Image.new("RGBA",(tile*2,42+(tile+24)*len(captures)),(3,8,14,255));draw=ImageDraw.Draw(canvas);draw.text((8,8),"CONTINUOUS CELL VAE // HELD-OUT",fill=(75,236,255,255));draw.text((8,27),"TARGET",fill=(184,255,73,255));draw.text((tile+8,27),"NEURAL CELL RASTER",fill=(184,255,73,255))
    for row,(identity,target,prediction) in enumerate(captures):
        y=42+row*(tile+24);canvas.alpha_composite(_image(target).resize((tile,tile),Image.Resampling.NEAREST),(0,y));canvas.alpha_composite(_image(prediction).resize((tile,tile),Image.Resampling.NEAREST),(tile,y));draw.text((8,y+tile+4),f"HELDOUT IDENTITY {identity:02d}",fill=(210,225,235,255))
    stream=io.BytesIO();canvas.save(stream,format="PNG",compress_level=9,optimize=False);visual=stream.getvalue();_atomic(root/SHEET_NAME,visual);gates={"all_values_finite":bool(np.isfinite(list(metrics.values())).all()),"alpha_iou_above_0_88":metrics["alpha_iou"]>.88,"rgba_mae_below_0_018":metrics["rgba_mae"]<.018,"foreground_rgb_mae_below_0_08":metrics["foreground_rgb_mae"]<.08};manifest={"format":EVALUATION_FORMAT,"status":"ready" if all(gates.values()) else "experimental","source_sha256":source_sha256(),"checkpoint":{"path":f"cell_vae_{contract['plan']['total_steps']:07d}.pt","sha256":sha256_file(root/f"cell_vae_{contract['plan']['total_steps']:07d}.pt"),"ema_state_sha256":checkpoint["ema_state_sha256"]},"evaluation":{"calibration_identity_count":5,"heldout_count":count,"metrics":metrics},"visual":{"path":SHEET_NAME,"bytes":len(visual),"sha256":hashlib.sha256(visual).hexdigest(),"visually_inspected":False},"gates":gates,"limitations":["Cell positions are supplied by the morphology and physics specialists.","The differentiable splat primitive remains deterministic scaffolding for later distillation."]};manifest["manifest_sha256"]=hashlib.sha256(canonical(manifest)).hexdigest();_atomic(root/MANIFEST_NAME,canonical(manifest));return manifest


def validate(root:Path=DEFAULT_OUTPUT)->dict[str,Any]:
    root=Path(root).resolve();raw=(root/MANIFEST_NAME).read_bytes();manifest=json.loads(raw)
    if raw!=canonical(manifest) or manifest.get("format")!=EVALUATION_FORMAT or manifest.get("source_sha256")!=source_sha256():raise ValueError("cell VAE evaluation drifted")
    expected=hashlib.sha256(canonical({k:v for k,v in manifest.items() if k!="manifest_sha256"})).hexdigest()
    if manifest.get("manifest_sha256")!=expected:raise ValueError("cell VAE manifest hash drifted")
    visual=root/manifest["visual"]["path"]
    if visual.stat().st_size!=manifest["visual"]["bytes"] or sha256_file(visual)!=manifest["visual"]["sha256"]:raise ValueError("cell VAE visual drifted")
    load_final(root);return {"passed":manifest["status"]=="ready","metrics":manifest["evaluation"]["metrics"],"manifest_sha256":manifest["manifest_sha256"]}
