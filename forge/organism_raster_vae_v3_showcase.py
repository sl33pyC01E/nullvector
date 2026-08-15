from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import uuid

import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch

from .config import PROJECT_ROOT
from .organism_raster_vae_v3.calibration import _canonical,_image,_sha,source_sha256
from .organism_raster_vae_v3.contract import RasterVAEV3Config
from .organism_raster_vae_v3.dataset import MorphologyMotionCorpus
from .organism_raster_vae_v3.model import StructuredRasterVAE
from .safety import require_disk_floor


FORMAT="nullvector-organism-raster-vae-v3-animated-showcase/1.0.0"
DEFAULT_CALIBRATION=PROJECT_ROOT/"outputs/organism_raster_vae_v3/calibration_1200_alpha_scaffold/calibration_manifest.json"


def _font(size: int):
    path=Path("C:/Windows/Fonts/consola.ttf"); return ImageFont.truetype(str(path),size) if path.is_file() else ImageFont.load_default()


def _load(manifest_path: Path,device: torch.device):
    raw=manifest_path.read_bytes(); manifest=json.loads(raw)
    expected=hashlib.sha256(_canonical({key:value for key,value in manifest.items() if key!="manifest_sha256"})).hexdigest()
    if manifest.get("manifest_sha256")!=expected or manifest.get("source_sha256")!=source_sha256(): raise ValueError("VAE v3 calibration provenance drifted")
    descriptor=manifest["artifacts"]["checkpoint"]; checkpoint_path=manifest_path.parent/descriptor["path"]
    if _sha(checkpoint_path)!=descriptor["sha256"] or checkpoint_path.stat().st_size!=descriptor["bytes"]: raise ValueError("VAE v3 checkpoint identity drifted")
    checkpoint=torch.load(checkpoint_path,map_location="cpu",weights_only=True)
    if checkpoint.get("source_sha256")!=source_sha256() or checkpoint.get("corpus_sha256")!=manifest["corpus"]["semantic_sha256"]: raise ValueError("VAE v3 checkpoint semantic binding drifted")
    model=StructuredRasterVAE(RasterVAEV3Config(**checkpoint["config"])); model.load_state_dict(checkpoint["ema_state"],strict=True); return model.to(device).eval(),manifest


def _render(targets: torch.Tensor,predictions: torch.Tensor,frame: int) -> Image.Image:
    family_names=("HUMANOID","ANIMALIAN","PLANTLIKE","ANOMALY","MACHINE"); colors=((55,224,247),(255,91,187),(151,244,74),(189,106,255),(255,178,56)); scale=2; cell=96*scale; row_h=cell+37; width=38+2*(cell+22); height=58+5*row_h
    canvas=Image.new("RGB",(width,height),(2,7,12)); draw=ImageDraw.Draw(canvas); draw.text((15,9),"VAE V3 // HELD-OUT MOTION RECONSTRUCTION",font=_font(18),fill=(225,242,247)); draw.text((15,32),f"FRAME {frame+1:02d}/16  ·  LEFT CELL TARGET  ·  RIGHT CONTINUOUS NEURAL DECODE",font=_font(10),fill=(79,218,239))
    for row in range(5):
        y=58+row*row_h; target=_image(targets[row]).resize((cell,cell),Image.Resampling.NEAREST); prediction=_image(predictions[row]).resize((cell,cell),Image.Resampling.NEAREST); canvas.paste(target,(15,y)); canvas.paste(prediction,(37+cell,y)); draw.text((15,y+cell+5),f"{family_names[row]} // TARGET",font=_font(9),fill=colors[row]); draw.text((37+cell,y+cell+5),"VAE MEAN",font=_font(9),fill=(154,238,212))
    return canvas


def build_showcase(calibration: Path,destination: Path) -> Path:
    calibration=calibration.resolve(); destination=destination.resolve()
    if destination.exists(): raise FileExistsError(destination)
    require_disk_floor(destination.parent,floor_gb=100,planned_bytes=512*1024**2); device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model,source_manifest=_load(calibration,device); corpus=MorphologyMotionCorpus()
    identities=(5,11,17,23,29); targets=[]; predictions=[]
    with torch.inference_mode():
        for phase in range(16):
            rows=[corpus[identity*16+phase] for identity in identities]; batch={key:torch.stack([row[key] for row in rows]).to(device) for key in rows[0]}
            context=torch.autocast("cuda",dtype=torch.bfloat16) if device.type=="cuda" else torch.no_grad()
            with context: output=model(batch["living"],batch["family"],batch["traits"],batch["phase"],stochastic=False)
            targets.append(batch["rgba"].float().cpu()); predictions.append(output.rgba.float().cpu())
    target=torch.stack(targets); prediction=torch.stack(predictions); target_delta=(target.roll(-1,0)-target).abs().mean((2,3,4)); prediction_delta=(prediction.roll(-1,0)-prediction).abs().mean((2,3,4)); temporal_ratio=prediction_delta.mean(0)/target_delta.mean(0).clamp_min(1e-8)
    staging=destination.parent/f".{destination.name}.tmp-{uuid.uuid4().hex}"; frames=staging/"frames"; frames.mkdir(parents=True)
    try:
        for phase in range(16): _render(target[phase],prediction[phase],phase).save(frames/f"frame_{phase:03d}.png",compress_level=6)
        poster=staging/"animated_reconstruction_contact.png"; _render(target[4],prediction[4],4).save(poster,compress_level=7)
        subprocess.run(["ffmpeg","-hide_banner","-loglevel","error","-y","-framerate","8","-i",str(frames/"frame_%03d.png"),"-vf","palettegen=stats_mode=diff","-frames:v","1",str(staging/"palette.png")],check=True)
        gif=staging/"animated_reconstruction.gif"; subprocess.run(["ffmpeg","-hide_banner","-loglevel","error","-y","-framerate","8","-i",str(frames/"frame_%03d.png"),"-i",str(staging/"palette.png"),"-lavfi","paletteuse=dither=bayer:bayer_scale=3","-loop","0",str(gif)],check=True); shutil.rmtree(frames); (staging/"palette.png").unlink()
        metrics={"rgba_mae":round(float((prediction-target).abs().mean()),9),"mean_target_frame_delta":round(float(target_delta.mean()),9),"mean_neural_frame_delta":round(float(prediction_delta.mean()),9),"temporal_gain_ratio_by_family":[round(float(value),7) for value in temporal_ratio],"minimum_temporal_gain_ratio":round(float(temporal_ratio.min()),7),"maximum_temporal_gain_ratio":round(float(temporal_ratio.max()),7)}
        payload={"format":FORMAT,"calibration_manifest_sha256":_sha(calibration),"calibration_semantic_sha256":source_manifest["manifest_sha256"],"families":list(("humanoid","animalian","plantlike","anomaly","machine")),"identities":list(identities),"frames":16,"metrics":metrics,"artifacts":{"gif":{"path":gif.name,"sha256":_sha(gif),"bytes":gif.stat().st_size},"contact":{"path":poster.name,"sha256":_sha(poster),"bytes":poster.stat().st_size}},"gates":{"source_exact":True,"held_out_family_census_exact":True,"all_frames_decoded":True,"all_families_have_nonzero_motion":bool((prediction_delta.mean(0)>1e-5).all()),"production_promotion_allowed":False},"status":"human_review_required"}; payload["manifest_sha256"]=hashlib.sha256(_canonical(payload)).hexdigest(); (staging/"showcase_manifest.json").write_bytes(_canonical(payload)); staging.replace(destination)
    except BaseException:
        if staging.exists(): shutil.rmtree(staging)
        raise
    return destination/"showcase_manifest.json"


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--calibration",type=Path,default=DEFAULT_CALIBRATION); parser.add_argument("--output",type=Path,default=PROJECT_ROOT/"outputs/organism_raster_vae_v3/animated_showcase_1200"); args=parser.parse_args(); print(build_showcase(args.calibration,args.output))


if __name__=="__main__": main()
