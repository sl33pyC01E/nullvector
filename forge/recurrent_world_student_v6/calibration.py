from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from ..recurrent_world_student_v5.model import PerceptionRecurrentWorldStudent
from ..world_action_natural_v10 import load
from ..world_latent_dit.contract import ModelConfig
from .contract import CHECKPOINT_FORMAT,DEFAULT_OUTPUT,PARENT,PARENT_SHA256,canonical,file_sha256,source_sha256
from .training import _normalizers


FORMAT="nullvector-natural-recurrent-trust-calibration-v6/1.0.0"
CALIBRATED_RUNTIME=DEFAULT_OUTPUT/"runtime_calibrated_ramp.pt"


def _calibration_source_sha256():return hashlib.sha256(b"nullvector-v6-trust-calibration\0"+Path(__file__).read_bytes()).hexdigest()


@torch.inference_mode()
def _score(model,sequence,norms,device,horizon,bias,ramp_steps=0,samples=48):
    lm,ls,am,ass=norms;starts=np.linspace(1,len(sequence["latent"])-horizon-1,min(samples,len(sequence["latent"])-horizon-1),dtype=np.int64);initial=torch.from_numpy(sequence["latent"][starts]).to(device);previous=torch.from_numpy(sequence["latent"][starts-1]).to(device);current=initial.clone();previous_actor=torch.from_numpy(sequence["actor_state"][starts-1]).to(device);actor=torch.from_numpy(sequence["actor_state"][starts]).to(device);gate_sum=0.
    for offset in range(horizon):
        indices=starts+offset;action=torch.from_numpy(sequence["action"][indices+1].astype(np.int64)).to(device);control=torch.from_numpy(sequence["control"][indices+1]).to(device);state=torch.from_numpy(sequence["state"][indices]).to(device);visibility=torch.from_numpy(sequence["visibility"][indices]).to(device);memory=torch.from_numpy(sequence["memory"][indices]).to(device);cn,pn=(current-lm)/ls,(previous-lm)/ls;delta,logits=model.gated_action(cn,pn,action,control,state,actor,visibility,memory);applied_bias=bias if ramp_steps<=0 else bias*min(offset/ramp_steps,1.);gate=torch.sigmoid(logits+applied_bias);next_latent=(cn+gate*delta)*ls+lm;an,pan=(actor-am)/ass,(previous_actor-am)/ass;result=model.actor(an,pan,action,control,state,visibility,memory);next_actor=(an+.9*(result.gate>=.7)*(result.state-an))*ass+am;previous,current=current,next_latent;previous_actor,actor=actor,next_actor;gate_sum+=float(gate.mean())
    target=torch.from_numpy(sequence["latent"][starts+horizon]).to(device);mae=float(F.l1_loss(current,target));persistence=float(F.l1_loss(initial,target));motion=float(F.l1_loss(current,initial));return {"horizon":horizon,"samples":len(starts),"mae":mae,"persistence_mae":persistence,"improvement":1-mae/persistence,"motion_ratio":motion/persistence,"mean_trust":gate_sum/horizon}


def _model(payload,device):
    model=PerceptionRecurrentWorldStudent(ModelConfig(**payload["model_config"]));model.load_state_dict(payload["state"]);return model.to(device).eval()


def calibrate(output:Path=DEFAULT_OUTPUT):
    output=Path(output).resolve();runtime=output/"runtime.pt";runtime_sha=file_sha256(runtime);payload=torch.load(runtime,map_location="cpu",weights_only=True)
    if payload.get("format")!=CHECKPOINT_FORMAT or payload.get("source_sha256")!=source_sha256():raise ValueError("V6 trust calibration runtime drifted")
    if file_sha256(PARENT)!=PARENT_SHA256:raise ValueError("V6 trust calibration parent drifted")
    device=torch.device("cuda:0");torch.cuda.set_per_process_memory_fraction(.45,0);sequences,manifest=load();model=_model(payload,device);norms=_normalizers(payload,device);biases=(1.,1.5,2.,2.5,3.);ramps=(2,4,8,16);horizons=(1,2,4,8,16,32);schedules=tuple((bias,ramp) for bias in biases for ramp in ramps);validation={}
    for bias,ramp in schedules:validation[f"{bias}/{ramp}"]={str(h):_score(model,sequences[4],norms,device,h,bias,ramp) for h in horizons}
    eligible=[(bias,ramp) for bias,ramp in schedules if validation[f"{bias}/{ramp}"]["32"]["motion_ratio"]>=.25 and all(validation[f"{bias}/{ramp}"][str(h)]["improvement"]>0 for h in horizons)]
    if not eligible:raise RuntimeError("no V6 trust bias passes validation motion and accuracy gates")
    chosen_bias,chosen_ramp=min(eligible,key=lambda schedule:sum(validation[f"{schedule[0]}/{schedule[1]}"][str(h)]["mae"]/validation[f"{schedule[0]}/{schedule[1]}"][str(h)]["persistence_mae"] for h in horizons));chosen_key=f"{chosen_bias}/{chosen_ramp}";test={str(h):_score(model,sequences[5],norms,device,h,chosen_bias,chosen_ramp) for h in horizons};parent_payload=torch.load(PARENT,map_location="cpu",weights_only=True);parent=_model(parent_payload,device);parent_norms=_normalizers(parent_payload,device);parent_test={str(h):_score(parent,sequences[5],parent_norms,device,h,0.) for h in horizons};candidate_vs_parent={str(h):1-test[str(h)]["mae"]/parent_test[str(h)]["mae"] for h in horizons};gates={"validation_motion_floor":validation[chosen_key]["32"]["motion_ratio"]>=.25,"all_test_horizons_beat_persistence":all(row["improvement"]>0 for row in test.values()),"all_test_horizons_beat_v5_parent":all(value>0 for value in candidate_vs_parent.values())};gates["all_passed"]=all(gates.values());calibrated=dict(payload);calibrated["inference"]={"gate_logit_bias_max":chosen_bias,"gate_logit_bias_ramp_steps":chosen_ramp,"selection":"validation-only multi-horizon ramp accuracy with horizon-32 motion floor 0.25","calibration_source_sha256":_calibration_source_sha256()};temporary=output/".runtime_calibrated_ramp.pt.tmp";torch.save(calibrated,temporary);os.replace(temporary,CALIBRATED_RUNTIME);report={"format":FORMAT,"calibration_source_sha256":_calibration_source_sha256(),"model_source_sha256":source_sha256(),"corpus_sha256":manifest["manifest_sha256"],"checkpoint_sha256":runtime_sha,"biases":list(biases),"ramps":list(ramps),"horizons":list(horizons),"validation":validation,"chosen":{"bias_max":chosen_bias,"ramp_steps":chosen_ramp},"test":test,"v5_parent_test":parent_test,"candidate_vs_v5_parent":candidate_vs_parent,"gates":gates,"calibrated_runtime_sha256":file_sha256(CALIBRATED_RUNTIME)};report["report_sha256"]=hashlib.sha256(canonical(report)).hexdigest();(output/"trust_calibration_ramp.json").write_bytes(canonical(report));return report


if __name__=="__main__":print(json.dumps(calibrate(),indent=2))
