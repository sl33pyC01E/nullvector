from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import uuid

import numpy as np
from PIL import Image, ImageDraw
import torch

from .contract import ModelConfig, canonical, source_sha256
from .data import load_corpus, rows
from .decoder import load_decoder
from .model import WholeViewportLatentModel
from .training import _prepare_latents, _tensor


@torch.inference_mode()
def render(*, release: Path, corpus: Path, output: Path, frames=96, device="cuda", decoder_release: Path | None = None):
    release, corpus, output = Path(release), Path(corpus), Path(output)
    manifest_raw=(release/"manifest.json").read_bytes();manifest=json.loads(manifest_raw)
    if manifest_raw != canonical(manifest) or manifest.get("source_sha256") != source_sha256():
        raise ValueError("whole-viewport release provenance drifted")
    artifact=release/manifest["artifact"]["path"]
    if artifact.stat().st_size != manifest["artifact"]["bytes"] or hashlib.sha256(artifact.read_bytes()).hexdigest() != manifest["artifact"]["sha256"]:
        raise ValueError("whole-viewport checkpoint drifted")
    target=torch.device(device if device=="cpu" or torch.cuda.is_available() else "cpu")
    payload=torch.load(artifact,map_location=target,weights_only=False)
    model=WholeViewportLatentModel(ModelConfig(**payload["model_config"])).to(target).eval();model.load_state_dict(payload["state"])
    decoder,decoder_provenance=load_decoder(target,decoder_release)
    if manifest.get("decoder") != decoder_provenance:
        raise ValueError("showcase decoder does not match release")
    episodes,_=load_corpus(corpus);data=_prepare_latents(decoder,rows(episodes[-1:]),target);count=min(int(frames),len(data["frame"]));previous=_tensor({"value":data["previous_latent"][:1]},"value",target)
    staging=output.parent/f".{output.name}.tmp-{uuid.uuid4().hex}";staging.mkdir(parents=True)
    generated=[]
    for index in range(count):
        chosen=np.asarray([index])
        with torch.autocast(target.type,dtype=torch.bfloat16,enabled=target.type=="cuda"):
            previous=model(previous,_tensor({"value":data["spatial"][chosen]},"value",target),_tensor({"value":data["organisms"][chosen]},"value",target),_tensor({"value":data["organism_mask"][chosen]},"value",target,torch.bool),_tensor({"value":data["state"][chosen]},"value",target),_tensor({"value":data["actor_state"][chosen]},"value",target),_tensor({"value":data["actor_field"][chosen]},"value",target),_tensor({"value":data["visibility"][chosen]},"value",target),_tensor({"value":data["memory"][chosen]},"value",target),_tensor({"value":data["control"][chosen]},"value",target),_tensor({"value":data["action"][chosen]},"value",target,torch.long));decoded=decoder.decode(previous).float().clamp(0,1)
        neural=(decoded[0].permute(1,2,0).cpu().numpy()*255+.5).astype(np.uint8);teacher=data["frame"][index];canvas=Image.new("RGB",(512,276),(4,10,13));canvas.paste(Image.fromarray(teacher),(0,20));canvas.paste(Image.fromarray(neural),(256,20));draw=ImageDraw.Draw(canvas);draw.text((8,4),f"TEACHER  F{index:04}",fill=(112,231,224));draw.text((264,4),"AUTOREGRESSIVE VAE VIEW",fill=(255,104,192));path=staging/f"frame_{index:04}.png";canvas.save(path,optimize=True);generated.append(neural)
    command=["ffmpeg","-y","-hide_banner","-loglevel","error","-framerate","12","-i",str(staging/"frame_%04d.png"),"-vf","fps=12,scale=1024:-1:flags=neighbor,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3","-loop","0",str(staging/"rollout.gif")]
    subprocess.run(command,check=True)
    indices=np.linspace(0,count-1,12).round().astype(int);contact=Image.new("RGB",(1024,1656),(4,10,13))
    for slot,index in enumerate(indices):contact.paste(Image.open(staging/f"frame_{index:04}.png"),(slot%2*512,slot//2*276))
    contact.save(staging/"contact.png",optimize=True)
    report={"format":"nullvector-whole-viewport-showcase/1.0.0","release_manifest_sha256":manifest["manifest_sha256"],"source_sha256":source_sha256(),"frames":count,"autoregressive":True,"gif":{"path":"rollout.gif","bytes":(staging/"rollout.gif").stat().st_size,"sha256":hashlib.sha256((staging/"rollout.gif").read_bytes()).hexdigest()},"contact":{"path":"contact.png","bytes":(staging/"contact.png").stat().st_size,"sha256":hashlib.sha256((staging/"contact.png").read_bytes()).hexdigest()}}
    report["report_sha256"]=hashlib.sha256(canonical(report)).hexdigest();(staging/"showcase.json").write_bytes(canonical(report))
    for path in staging.glob("frame_*.png"):path.unlink()
    if output.exists():raise FileExistsError(output)
    staging.replace(output);return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--release",type=Path,required=True);parser.add_argument("--corpus",type=Path,required=True);parser.add_argument("--output",type=Path,required=True);parser.add_argument("--decoder-release",type=Path);parser.add_argument("--frames",type=int,default=96);parser.add_argument("--device",default="cuda");args=parser.parse_args();print(json.dumps(render(release=args.release,corpus=args.corpus,output=args.output,frames=args.frames,device=args.device,decoder_release=args.decoder_release),indent=2))


if __name__=="__main__":main()
