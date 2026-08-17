from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import json
from threading import Lock

import numpy as np
import torch

from ..recurrent_world_rollout_v1.contract import source_sha256 as rollout_source_sha256
from ..recurrent_world_student_v3.contract import CHECKPOINT_FORMAT,state_sha256
from ..recurrent_world_student_v3.model import RecurrentWorldStudent
from ..world_frame_decoder_adapt_v1 import AdaptedWorldFrameCodec
from ..world_latent_dit.contract import LATENT_CHANNELS,ModelConfig
from .contract import CODEC_CHECKPOINT,CODEC_SHA256,FORMAT,ROLLOUT_REPORT,V3_CHECKPOINT,V3_REPORT,V3_SHA256,canonical,file_sha256


@dataclass(frozen=True,slots=True)
class WorldStep:
    frame:np.ndarray
    actor_state:np.ndarray
    index:int


@dataclass(frozen=True,slots=True)
class WorldForecast:
    frames:np.ndarray
    actor_states:np.ndarray

    @property
    def final_frame(self)->np.ndarray:return self.frames[-1]

    @property
    def final_actor_state(self)->np.ndarray:return self.actor_states[-1]


def _canonical_report(path):
    raw=path.read_bytes();payload=json.loads(raw)
    if raw!=canonical(payload):raise ValueError(f"non-canonical recurrent world report: {path.name}")
    return payload


def _validate_release():
    production=_canonical_report(V3_REPORT);rollout=_canonical_report(ROLLOUT_REPORT)
    if production.get("status")!="ready" or production.get("checkpoint",{}).get("sha256")!=V3_SHA256:raise ValueError("recurrent V3 production release is not ready")
    if file_sha256(V3_CHECKPOINT)!=V3_SHA256:raise ValueError("recurrent V3 checkpoint drifted")
    if rollout.get("source_sha256")!=rollout_source_sha256() or rollout.get("candidate_sha256")!=V3_SHA256 or rollout.get("status")!="long_horizon_ready" or not rollout.get("gates",{}).get("all_passed"):raise ValueError("recurrent V3 long-horizon authority drifted")
    digest=rollout.get("report_sha256");without=dict(rollout);without.pop("report_sha256",None)
    if digest!=hashlib.sha256(canonical(without)).hexdigest():raise ValueError("recurrent rollout report hash drifted")
    if production.get("corpus_sha256")!=rollout.get("corpus_sha256"):raise ValueError("recurrent rollout corpus drifted")
    if file_sha256(CODEC_CHECKPOINT)!=CODEC_SHA256:raise ValueError("adapted world codec drifted")
    return production,rollout


class RecurrentWorldStream:
    def __init__(self,runtime,current_frame,actor_state,*,previous_frame=None,previous_actor_state=None):
        self.runtime=runtime;self.index=0
        current=runtime._frame(current_frame);previous=current if previous_frame is None else runtime._frame(previous_frame)
        actor=runtime._vector(actor_state,128,"actor_state");previous_actor=actor if previous_actor_state is None else runtime._vector(previous_actor_state,128,"previous_actor_state")
        with runtime.lock,torch.inference_mode():
            self.current=runtime.codec.encode(current);self.previous=self.current.clone() if previous_frame is None else runtime.codec.encode(previous)
            self.actor=torch.from_numpy(actor)[None].to(runtime.device);self.previous_actor=torch.from_numpy(previous_actor)[None].to(runtime.device)
            self.visual=torch.from_numpy(current.copy()).permute(2,0,1)[None].float().div_(255).to(runtime.device)
            self.decoded=runtime.codec.model.decode(self.current)

    def advance(self,action,control,state)->WorldStep:
        runtime=self.runtime;action_value=int(action)
        if not 0<=action_value<22:raise ValueError("action index must be in [0,22)")
        control_value=runtime._vector(control,4,"control");state_value=runtime._vector(state,64,"state")
        action_tensor=torch.tensor((action_value,),dtype=torch.long,device=runtime.device);control_tensor=torch.from_numpy(control_value)[None].to(runtime.device);state_tensor=torch.from_numpy(state_value)[None].to(runtime.device)
        context=torch.autocast("cuda",dtype=torch.bfloat16) if runtime.device.type=="cuda" else nullcontext()
        with runtime.lock,torch.inference_mode(),context:
            cn=(self.current-runtime.latent_mean)/runtime.latent_std;pn=(self.previous-runtime.latent_mean)/runtime.latent_std
            delta=runtime.model.action(cn,pn,action_tensor,control_tensor,state_tensor,self.actor);next_latent=(cn+(delta.abs().mean(1,keepdim=True)>=runtime.latent_threshold)*delta)*runtime.latent_std+runtime.latent_mean
            an=(self.actor-runtime.actor_mean)/runtime.actor_std;pan=(self.previous_actor-runtime.actor_mean)/runtime.actor_std;actor_result=runtime.model.actor(an,pan,action_tensor,control_tensor,state_tensor);next_actor=(an+runtime.actor_alpha*(actor_result.gate>=runtime.actor_threshold)*(actor_result.state-an))*runtime.actor_std+runtime.actor_mean
            next_decoded=runtime.codec.model.decode(next_latent);self.visual=torch.clamp(self.visual+next_decoded-self.decoded,0,1)
            self.previous,self.current=self.current,next_latent;self.previous_actor,self.actor=self.actor,next_actor;self.decoded=next_decoded;self.index+=1
            frame=self.visual[0].permute(1,2,0).float().mul(255).clamp(0,255).to(torch.uint8).cpu().numpy();actor=self.actor[0].float().cpu().numpy()
        return WorldStep(frame,actor,self.index)


class RecurrentWorldRuntime:
    """Validated continuous frame+actor recurrent runtime.

    The model predicts latent visual change and causal actor state. The adapted
    VAE renders only the predicted residual, preserving unchanged live pixels.
    """

    latent_threshold=.18
    actor_threshold=.7
    actor_alpha=.9

    def __init__(self,model,codec,device,normalization,production_report,rollout_report):
        self.model=model.eval();self.codec=codec;self.device=device;self.lock=Lock();self.production_report=production_report;self.rollout_report=rollout_report
        self.latent_mean=torch.tensor(normalization["latent_mean"],device=device)[None,:,None,None];self.latent_std=torch.tensor(normalization["latent_std"],device=device)[None,:,None,None];self.actor_mean=torch.tensor(normalization["actor_mean"],device=device)[None];self.actor_std=torch.tensor(normalization["actor_std"],device=device)[None]
        if self.latent_mean.shape!=(1,LATENT_CHANNELS,1,1) or self.actor_mean.shape!=(1,128) or not torch.all(self.latent_std>0) or not torch.all(self.actor_std>0):raise ValueError("recurrent world normalization drifted")

    @classmethod
    def from_release(cls,*,device="cuda"):
        production,rollout=_validate_release();payload=torch.load(V3_CHECKPOINT,map_location="cpu",weights_only=True)
        if payload.get("format")!=CHECKPOINT_FORMAT or payload.get("source_sha256")!=production.get("source_sha256") or payload.get("corpus_sha256")!=production.get("corpus_sha256") or state_sha256(payload["state"])!=payload.get("state_sha256"):raise ValueError("recurrent V3 runtime state drifted")
        target=torch.device(device if device!="cuda" or torch.cuda.is_available() else "cpu");model=RecurrentWorldStudent(ModelConfig(**payload["model_config"]));model.load_state_dict(payload["state"],strict=True);model.to(target);codec=AdaptedWorldFrameCodec.from_checkpoint(CODEC_CHECKPOINT,device=str(target));return cls(model,codec,target,payload["normalization"],production,rollout)

    @staticmethod
    def _frame(value):
        result=np.asarray(value)
        if result.shape!=(256,256,3) or result.dtype!=np.uint8:raise ValueError("world frame must be uint8 HWC RGB 256x256")
        return np.ascontiguousarray(result)

    @staticmethod
    def _vector(value,width,name):
        result=np.asarray(value,dtype=np.float32)
        if result.shape!=(width,) or not np.isfinite(result).all():raise ValueError(f"{name} must be finite float32 [{width}]")
        return np.ascontiguousarray(result)

    def stream(self,current_frame,actor_state,*,previous_frame=None,previous_actor_state=None):
        return RecurrentWorldStream(self,current_frame,actor_state,previous_frame=previous_frame,previous_actor_state=previous_actor_state)

    def forecast(self,current_frame,actor_state,*,actions,controls,states,previous_frame=None,previous_actor_state=None,horizon=None)->WorldForecast:
        action_values=np.asarray(actions,dtype=np.int64).reshape(-1);count=len(action_values) if horizon is None else int(horizon)
        if not 1<=count<=32:raise ValueError("forecast horizon must be in [1,32]")
        if len(action_values)==1:action_values=np.repeat(action_values,count)
        if len(action_values)!=count:raise ValueError("action schedule length drifted")
        def schedule(value,width,name):
            array=np.asarray(value,dtype=np.float32)
            if array.shape==(width,):array=np.repeat(array[None],count,axis=0)
            if array.shape!=(count,width) or not np.isfinite(array).all():raise ValueError(f"{name} schedule must be [{count},{width}]")
            return array
        control_values=schedule(controls,4,"control");state_values=schedule(states,64,"state");stream=self.stream(current_frame,actor_state,previous_frame=previous_frame,previous_actor_state=previous_actor_state);steps=[stream.advance(action_values[index],control_values[index],state_values[index]) for index in range(count)];return WorldForecast(np.stack([step.frame for step in steps]),np.stack([step.actor_state for step in steps]))

    @property
    def parameter_count(self):return self.model.parameter_count

    @property
    def authority(self):return {"format":FORMAT,"checkpoint_sha256":V3_SHA256,"long_horizon_status":self.rollout_report["status"],"horizons":tuple(int(value) for value in self.rollout_report["horizons"])}
