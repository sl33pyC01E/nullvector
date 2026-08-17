from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
from PIL import Image,ImageDraw
import torch

from .cache import load as load_cache
from .contract import DEFAULT_OUTPUT,MANIFEST_FORMAT,canonical,sha256_file,source_sha256
from .training import load_final


MANIFEST_NAME="evaluation_manifest.json";SHEET_NAME="heldout_neural_refinement.png"


def _atomic(path:Path,value:bytes)->None:
    descriptor,name=tempfile.mkstemp(prefix=f".{path.name}.tmp-",dir=path.parent);temporary=Path(name)
    try:
        with os.fdopen(descriptor,"wb") as handle:handle.write(value);handle.flush();os.fsync(handle.fileno())
        os.replace(temporary,path)
    finally:temporary.unlink(missing_ok=True)


def _image(value:torch.Tensor)->Image.Image:
    array=(value.detach().cpu().permute(1,2,0).numpy().clip(0,1)*255+.5).astype(np.uint8);return Image.fromarray(array,"RGBA")


@torch.inference_mode()
def evaluate(root:Path=DEFAULT_OUTPUT,*,device_name:str="cuda")->dict[str,Any]:
    root=Path(root).resolve();path=root/MANIFEST_NAME
    if path.exists():raise FileExistsError("V7 evaluation is immutable")
    cache=load_cache(root);model,checkpoint,contract=load_final(root);device=torch.device(device_name);model.to(device)
    indices=[i for i,v in enumerate(cache["identity"].tolist()) if int(v) in {5,11,17,23,29}];sums={name:0. for name in ("parent_alpha_iou","refined_alpha_iou","parent_rgba_mae","refined_rgba_mae","parent_appendage_recall","refined_appendage_recall")};count=0;captures=[]
    for start in range(0,len(indices),8):
        chosen=torch.tensor(indices[start:start+8]);living=cache["living"][chosen].to(device);parent=cache["parent_rgba"][chosen].to(device).float();target=cache["target_rgba"][chosen].to(device).float();limb=cache["appendage_alpha"][chosen].to(device)>.5;refined=model(living,parent).rgba.float();alpha=target[:,3:]>.5
        for label,prediction in (("parent",parent),("refined",refined)):
            pa=prediction[:,3:]>.5;inter=(pa&alpha).flatten(1).sum(1).float();union=(pa|alpha).flatten(1).sum(1).float().clamp_min(1);recall=(pa&limb).flatten(1).sum(1).float()/limb.flatten(1).sum(1).float().clamp_min(1);sums[f"{label}_alpha_iou"]+=float((inter/union).sum());sums[f"{label}_rgba_mae"]+=float((prediction-target).abs().flatten(1).mean(1).sum());sums[f"{label}_appendage_recall"]+=float(recall.sum())
        for local,index in enumerate(chosen.tolist()):
            if int(cache["phase_index"][index])==8:captures.append((int(cache["identity"][index]),target[local].cpu(),parent[local].cpu(),refined[local].cpu()))
        count+=len(chosen)
    metrics={k:round(v/count,8) for k,v in sums.items()};tile=192;canvas=Image.new("RGBA",(tile*3,42+(tile+24)*len(captures)),(3,8,14,255));draw=ImageDraw.Draw(canvas);draw.text((10,10),"CURRENT-CORPUS ANATOMICAL VAE // HELD-OUT NEURAL CELL REFINEMENT",fill=(75,236,255,255))
    for column,title in enumerate(("TARGET","V6 PARENT","V7 REFINED")):draw.text((column*tile+8,28),title,fill=(184,255,73,255))
    for row,(identity,target,parent,refined) in enumerate(captures):
        y=42+row*(tile+24)
        for column,value in enumerate((target,parent,refined)):canvas.alpha_composite(_image(value).resize((tile,tile),Image.Resampling.NEAREST),(column*tile,y))
        draw.text((8,y+tile+4),f"HELDOUT IDENTITY {identity:02d}",fill=(210,225,235,255))
    stream=io.BytesIO();canvas.save(stream,format="PNG",compress_level=9,optimize=False);visual=stream.getvalue();_atomic(root/SHEET_NAME,visual)
    gates={"all_values_finite":bool(np.isfinite(list(metrics.values())).all()),"alpha_iou_above_0_82":metrics["refined_alpha_iou"]>.82,"alpha_iou_improves_parent_0_15":metrics["refined_alpha_iou"]>metrics["parent_alpha_iou"]+.15,"rgba_mae_below_0_02":metrics["refined_rgba_mae"]<.02,"appendage_recall_above_0_92":metrics["refined_appendage_recall"]>.92}
    manifest={"format":MANIFEST_FORMAT,"status":"ready" if all(gates.values()) else "experimental","source_sha256":source_sha256(),"checkpoint":{"path":f"refiner_{contract['plan']['total_steps']:07d}.pt","sha256":sha256_file(root/f"refiner_{contract['plan']['total_steps']:07d}.pt"),"ema_state_sha256":checkpoint["ema_state_sha256"]},"evaluation":{"heldout_count":count,"metrics":metrics},"visual":{"path":SHEET_NAME,"bytes":len(visual),"sha256":hashlib.sha256(visual).hexdigest(),"visually_inspected":False},"gates":gates,"limitations":["The V7 refiner consumes the V6 VAE render and anatomical cell field.","It is not yet distilled into a single mobile decoder."]};manifest["manifest_sha256"]=hashlib.sha256(canonical(manifest)).hexdigest();_atomic(path,canonical(manifest));return manifest


def validate(root:Path=DEFAULT_OUTPUT)->dict[str,Any]:
    root=Path(root).resolve();raw=(root/MANIFEST_NAME).read_bytes();manifest=json.loads(raw)
    if raw!=canonical(manifest) or manifest.get("format")!=MANIFEST_FORMAT or manifest.get("source_sha256")!=source_sha256():raise ValueError("V7 evaluation provenance drifted")
    expected=hashlib.sha256(canonical({k:v for k,v in manifest.items() if k!="manifest_sha256"})).hexdigest()
    if manifest.get("manifest_sha256")!=expected:raise ValueError("V7 evaluation hash drifted")
    visual=root/manifest["visual"]["path"]
    if visual.stat().st_size!=manifest["visual"]["bytes"] or sha256_file(visual)!=manifest["visual"]["sha256"]:raise ValueError("V7 visual drifted")
    load_final(root)
    return {"passed":manifest["status"]=="ready","metrics":manifest["evaluation"]["metrics"],"manifest_sha256":manifest["manifest_sha256"]}
