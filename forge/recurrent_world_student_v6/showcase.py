from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import uuid

import numpy as np
from PIL import Image,ImageDraw
import torch

from ..recurrent_world_student_v5.model import PerceptionRecurrentWorldStudent
from ..safety import require_disk_floor
from ..world_action_natural_v10 import load
from ..world_frame_decoder_adapt_v1 import AdaptedWorldFrameCodec
from ..world_latent_dit.contract import ModelConfig
from .contract import CHECKPOINT_FORMAT,CODEC,CODEC_SHA256,DEFAULT_OUTPUT,canonical,file_sha256,source_sha256
from .training import _normalizers


FORMAT="nullvector-natural-recurrent-rollout-showcase-v6/1.0.0"


def _showcase_source_sha256():
    return hashlib.sha256(b"nullvector-v6-showcase\0"+Path(__file__).read_bytes()).hexdigest()


@torch.inference_mode()
def _rollout(model,sequence,start,count,norms,device,gate_bias_max,gate_bias_ramp_steps):
    lm,ls,am,ass=norms;previous=torch.from_numpy(sequence["latent"][start-1:start]).to(device);current=torch.from_numpy(sequence["latent"][start:start+1]).to(device);previous_actor=torch.from_numpy(sequence["actor_state"][start-1:start]).to(device);actor=torch.from_numpy(sequence["actor_state"][start:start+1]).to(device);rows=[current.cpu()]
    for offset in range(count-1):
        index=start+offset;action=torch.from_numpy(sequence["action"][index+1:index+2].astype(np.int64)).to(device);control=torch.from_numpy(sequence["control"][index+1:index+2]).to(device);state=torch.from_numpy(sequence["state"][index:index+1]).to(device);visibility=torch.from_numpy(sequence["visibility"][index:index+1]).to(device);memory=torch.from_numpy(sequence["memory"][index:index+1]).to(device);cn,pn=(current-lm)/ls,(previous-lm)/ls
        with torch.autocast("cuda",dtype=torch.bfloat16):delta,logits=model.gated_action(cn,pn,action,control,state,actor,visibility,memory)
        applied_bias=gate_bias_max*min(offset/gate_bias_ramp_steps,1.) if gate_bias_ramp_steps>0 else gate_bias_max;next_latent=(cn+torch.sigmoid(logits+applied_bias)*delta)*ls+lm;an,pan=(actor-am)/ass,(previous_actor-am)/ass;result=model.actor(an,pan,action,control,state,visibility,memory);next_actor=(an+.9*(result.gate>=.7)*(result.state-an))*ass+am;previous,current=current,next_latent;previous_actor,actor=actor,next_actor;rows.append(current.float().cpu())
    return torch.cat(rows)


@torch.inference_mode()
def _decode(codec,latent):
    rows=[]
    for start in range(0,len(latent),8):rows.append(codec.model.decode(latent[start:start+8].to(codec.device)).float().cpu())
    value=torch.clamp(torch.cat(rows),0,1).permute(0,2,3,1).numpy();return np.rint(value*255).astype(np.uint8)


def _panel(truth,prediction,index):
    height,width=truth.shape[:2];image=Image.new("RGB",(width*2,height+24),(4,10,15));image.paste(Image.fromarray(truth),(0,24));image.paste(Image.fromarray(prediction),(width,24));draw=ImageDraw.Draw(image);draw.text((8,6),f"TEACHER  t+{index:02d}",fill=(120,235,255));draw.text((width+8,6),"V6 RECURRENT",fill=(190,255,105));draw.line((width,0,width,height+24),fill=(30,75,90));return image


def build(output:Path=DEFAULT_OUTPUT/"showcase",*,start=96,count=48,fps=10):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    require_disk_floor(output.parent,floor_gb=100,planned_bytes=512*1024**2)
    runtime_path=DEFAULT_OUTPUT/"runtime_calibrated_ramp.pt";runtime_sha=file_sha256(runtime_path)
    if file_sha256(CODEC)!=CODEC_SHA256:raise ValueError("V6 showcase codec drifted")
    payload=torch.load(runtime_path,map_location="cpu",weights_only=True)
    if payload.get("format")!=CHECKPOINT_FORMAT or payload.get("source_sha256")!=source_sha256():raise ValueError("V6 showcase runtime drifted")
    device=torch.device("cuda:0");torch.cuda.set_per_process_memory_fraction(.45,0);model=PerceptionRecurrentWorldStudent(ModelConfig(**payload["model_config"]));model.load_state_dict(payload["state"]);model.to(device).eval();codec=AdaptedWorldFrameCodec.from_checkpoint(CODEC,device="cuda");sequences,manifest=load();sequence=sequences[5]
    if start<1 or start+count>=len(sequence["latent"]):raise ValueError("V6 showcase range invalid")
    inference=payload.get("inference",{});gate_bias=float(inference.get("gate_logit_bias_max",0.));gate_ramp=int(inference.get("gate_logit_bias_ramp_steps",0));predicted=_decode(codec,_rollout(model,sequence,start,count,_normalizers(payload,device),device,gate_bias,gate_ramp));truth=sequence["frame"][start:start+count];staging=output.parent/f".{output.name}.tmp-{uuid.uuid4().hex}";frames=staging/"frames";frames.mkdir(parents=True)
    try:
        hashes=[]
        for index,(target,result) in enumerate(zip(truth,predicted,strict=True)):
            path=frames/f"frame_{index:04d}.png";_panel(target,result,index).save(path,optimize=False);hashes.append(file_sha256(path))
        sample_indices=np.linspace(0,count-1,8,dtype=np.int64);sample=[Image.open(frames/f"frame_{int(index):04d}.png").convert("RGB") for index in sample_indices];contact=Image.new("RGB",(sample[0].width*4,sample[0].height*2));
        for index,image in enumerate(sample):contact.paste(image,((index%4)*image.width,(index//4)*image.height))
        contact_path=staging/"heldout_rollout_contact.png";contact.save(contact_path,optimize=False)
        ffmpeg=shutil.which("ffmpeg")
        if not ffmpeg:raise RuntimeError("ffmpeg is required for V6 showcase")
        gif_path=staging/"heldout_rollout.gif";mp4_path=staging/"heldout_rollout.mp4"
        subprocess.run([ffmpeg,"-hide_banner","-loglevel","error","-y","-framerate",str(fps),"-i",str(frames/"frame_%04d.png"),"-vf","split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3",str(gif_path)],check=True)
        subprocess.run([ffmpeg,"-hide_banner","-loglevel","error","-y","-framerate",str(fps),"-i",str(frames/"frame_%04d.png"),"-c:v","libx264","-pix_fmt","yuv420p","-crf","18",str(mp4_path)],check=True)
        report={"format":FORMAT,"showcase_source_sha256":_showcase_source_sha256(),"model_source_sha256":source_sha256(),"checkpoint_sha256":runtime_sha,"checkpoint_state_sha256":payload["state_sha256"],"gate_logit_bias_max":gate_bias,"gate_logit_bias_ramp_steps":gate_ramp,"corpus_sha256":manifest["manifest_sha256"],"session_index":5,"start":start,"frames":count,"fps":fps,"frame_tree_sha256":hashlib.sha256("".join(hashes).encode()).hexdigest(),"artifacts":{name:{"path":path.name,"bytes":path.stat().st_size,"sha256":file_sha256(path)} for name,path in (("contact",contact_path),("gif",gif_path),("mp4",mp4_path))},"status":"decoded_pending_visual_inspection"};report["report_sha256"]=hashlib.sha256(canonical(report)).hexdigest();(staging/"showcase_report.json").write_bytes(canonical(report));shutil.rmtree(frames);os.replace(staging,output)
    except BaseException:shutil.rmtree(staging,ignore_errors=True);raise
    return report


if __name__=="__main__":print(json.dumps(build(),indent=2))
