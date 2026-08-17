from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image,ImageDraw
import torch

from ..recurrent_world_student_v3.contract import CORPUS
from ..world_action_contiguous_v8 import load
from .contract import DEFAULT_OUTPUT,FORMAT,canonical,file_sha256,source_sha256
from .runtime import RecurrentWorldRuntime


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--device",default="cuda");parser.add_argument("--horizon",type=int,default=8);parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT);args=parser.parse_args();runtime=RecurrentWorldRuntime.from_release(device=args.device);sequences,manifest=load(CORPUS);sample=sequences[5];start=1;horizon=args.horizon
    kwargs={"actions":sample["action"][start+1:start+horizon+1],"controls":sample["control"][start+1:start+horizon+1],"states":sample["state"][start:start+horizon],"previous_frame":sample["frame"][start-1],"previous_actor_state":sample["actor_state"][start-1]}
    if runtime.device.type=="cuda":torch.cuda.reset_peak_memory_stats(runtime.device);torch.cuda.synchronize(runtime.device)
    began=time.perf_counter();result=runtime.forecast(sample["frame"][start],sample["actor_state"][start],**kwargs)
    if runtime.device.type=="cuda":torch.cuda.synchronize(runtime.device)
    elapsed=time.perf_counter()-began;repeat=runtime.forecast(sample["frame"][start],sample["actor_state"][start],**kwargs);exact=bool(np.array_equal(result.frames,repeat.frames) and np.array_equal(result.actor_states,repeat.actor_states));output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    marks=tuple(sorted(set((1,min(2,horizon),min(4,horizon),horizon))));sheet=Image.new("RGB",(256*len(marks),548),(3,9,13));draw=ImageDraw.Draw(sheet)
    for column,step in enumerate(marks):
        x=column*256;sheet.paste(Image.fromarray(result.frames[step-1]),(x,18));sheet.paste(Image.fromarray(sample["frame"][start+step]),(x,292));draw.text((x+7,4),f"PREDICTED +{step}",fill=(81,229,255));draw.text((x+7,278),f"AUTHORITY +{step}",fill=(178,255,103))
    comparison=output/"forecast_comparison.png";sheet.save(comparison,optimize=True)
    artifact={"path":comparison.name,"bytes":comparison.stat().st_size,"sha256":file_sha256(comparison)};report={"format":FORMAT,"status":"runtime_ready_legacy_target" if exact else "nondeterministic","source_sha256":source_sha256(),"corpus_sha256":manifest["manifest_sha256"],"device":str(runtime.device),"horizon":horizon,"seconds":round(elapsed,6),"steps_per_second":round(horizon/elapsed,4),"parameters":runtime.parameter_count,"peak_reserved_bytes":int(torch.cuda.max_memory_reserved(runtime.device)) if runtime.device.type=="cuda" else 0,"exact_replay":exact,"frame_sha256":hashlib.sha256(result.frames.tobytes()).hexdigest(),"actor_sha256":hashlib.sha256(result.actor_states.tobytes()).hexdigest(),"comparison":artifact,"visual_audit":{"clean_student_target":False,"legacy_diagnostic_overlays_detected":True,"disposition":"callable diagnostic; excluded from clean student rendering pending clean-corpus retraining"},"authority":runtime.authority};report["report_sha256"]=hashlib.sha256(canonical(report)).hexdigest();(output/"report.json").write_bytes(canonical(report));print(json.dumps(report,indent=2))


if __name__=="__main__":main()
