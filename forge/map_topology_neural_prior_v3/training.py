from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import torch

from ..config import PROJECT_ROOT
from ..map_topology_neural.codec import build_codec
from ..map_topology_neural_prior_corpus.shard import _load_arrays
from ..map_topology_neural_prior_generation.contract import CODEC_CHECKPOINT_RELATIVE
from ..map_topology_neural_prior_training.contract import FROZEN_LATENT_CORPUS_RELATIVE
from ..map_topology_neural_prior_training.dataset import PriorTrainingDataset
from ..map_topology_neural_prior_v2.masking import mask_tokens_v2
from ..map_topology_neural_prior_v2.model import build_prior_v2
from ..map_topology_neural_prior_v2_training.checkpoint import load_checkpoint as load_v2_checkpoint
from ..map_topology_neural_prior_v2_training.contract import PriorV2CalibrationConfig, canonical_json_bytes, sha256_file
from ..map_topology_neural_production.checkpoint import load_checkpoint as load_codec_checkpoint, tensor_state_sha256
from ..map_topology_neural_production.contract import TopologyCodecCalibrationConfig
from ..safety import require_disk_floor
from .semantic import frequency_weights, semantic_token_tables, semantic_topology_loss


FORMAT="nullvector-neural-map-topology-prior-v3-semantic-checkpoint/1.0.0"


def _source_sha256() -> str:
    paths=(Path(__file__).with_name("semantic.py"),Path(__file__))
    return hashlib.sha256(canonical_json_bytes({path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})).hexdigest()


def _token_counts(dataset: PriorTrainingDataset) -> torch.Tensor:
    counts=torch.zeros(512,dtype=torch.long)
    for shard_id in sorted({ref.shard_id for ref in dataset.refs_by_split["train"]}):
        tokens=torch.from_numpy(_load_arrays(dataset.latent_root/"shards"/shard_id/"latents.npz")["tokens"].astype("int64"))
        counts+=torch.bincount(tokens.flatten(),minlength=512)
    return counts


def load_checkpoint(path: Path) -> dict[str,Any]:
    path=Path(path).resolve()
    if not path.is_file() or path.stat().st_size>512*1024*1024:raise ValueError("Prior-v3 checkpoint is missing or oversized.")
    payload=torch.load(path,map_location="cpu",weights_only=True)
    required={"format","source_sha256","base_checkpoint_sha256","config","base_step","semantic_step","model_state","ema_state","optimizer_state","generator_state","history","model_sha256","ema_sha256","token_weights","walkable_table","hazard_table"}
    if not isinstance(payload,dict) or set(payload)!=required or payload["format"]!=FORMAT or payload["source_sha256"]!=_source_sha256():raise ValueError("Prior-v3 checkpoint contract drifted.")
    PriorV2CalibrationConfig.from_dict(payload["config"])
    if tensor_state_sha256(payload["model_state"])!=payload["model_sha256"] or tensor_state_sha256(payload["ema_state"])!=payload["ema_sha256"]:raise ValueError("Prior-v3 tensor identity failed.")
    if payload["semantic_step"]!=len(payload["history"]):raise ValueError("Prior-v3 history drifted.")
    return payload


def train_segment(base: Path, output: Path, *, updates: int=100, device_name: str="cuda") -> dict[str,Any]:
    output=Path(output).resolve()
    if output.exists():raise FileExistsError("Prior-v3 segment output is immutable.")
    if not 1<=updates<=500:raise ValueError("Prior-v3 segment updates must be in [1,500].")
    require_disk_floor(output.parent,floor_gb=100,planned_bytes=768*1024*1024)
    base_path=Path(base).resolve();raw=torch.load(base_path,map_location="cpu",weights_only=True)
    if raw.get("format")==FORMAT:
        previous=load_checkpoint(base_path);config=PriorV2CalibrationConfig.from_dict(previous["config"]);base_step=int(previous["base_step"]);semantic_start=int(previous["semantic_step"]);model_state=previous["model_state"];ema=previous["ema_state"]
    else:
        previous=load_v2_checkpoint(base_path);config=PriorV2CalibrationConfig.from_dict(previous["config"]);base_step=int(previous["step"]);semantic_start=0;model_state=previous["ema_state"];ema=previous["ema_state"]
    device=torch.device("cuda" if device_name=="cuda" and torch.cuda.is_available() else "cpu");torch.manual_seed(config.seed^0x563353454D414E54)
    dataset=PriorTrainingDataset(PROJECT_ROOT/"outputs/map_decorator_corpus_v1",PROJECT_ROOT/FROZEN_LATENT_CORPUS_RELATIVE);counts=_token_counts(dataset);weights=frequency_weights(counts)
    codec_payload=load_codec_checkpoint(PROJECT_ROOT/CODEC_CHECKPOINT_RELATIVE);codec_config=TopologyCodecCalibrationConfig.from_dict(codec_payload["config"]);codec=build_codec(codec_config.codec_config(),init_seed=codec_config.seed);codec.load_state_dict(codec_payload["ema_state"],strict=True);codec.eval();walkable,hazard=semantic_token_tables(codec);codec=codec.to(device)
    for parameter in codec.parameters():parameter.requires_grad_(False)
    model=build_prior_v2(config.model_config()).to(device);model.load_state_dict(model_state,strict=True);ema={name:value.to(device) for name,value in ema.items()};optimizer=torch.optim.AdamW(model.parameters(),lr=8e-5,weight_decay=config.weight_decay)
    generator=torch.Generator(device="cpu").manual_seed(config.seed^0x5633545241494E)
    if raw.get("format")==FORMAT:
        optimizer.load_state_dict(previous["optimizer_state"])
        for state in optimizer.state.values():
            for name,value in state.items():
                if isinstance(value,torch.Tensor):state[name]=value.to(device)
        generator.set_state(previous["generator_state"])
    history=[] if semantic_start==0 else list(previous["history"]);started=time.perf_counter();model.train()
    for local in range(updates):
        step=semantic_start+local;refs=dataset.training_refs(base_step+step,generator,config);batch=dataset.collate(refs);masked=mask_tokens_v2(batch["targets"],batch["valid_mask"],generator=generator,config=config.model_config(),step=base_step+step);inputs={name:batch[name].to(device) for name in ("valid_mask","point_conditions","global_conditions","theme_index")};inputs.update(tokens=masked["tokens"].to(device),mask_fraction=masked["mask_fraction"].to(device));optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda",dtype=torch.bfloat16,enabled=device.type=="cuda"):logits=model(inputs)
        terms=semantic_topology_loss(logits.float(),batch["targets"].to(device),masked["mask"].to(device),inputs["valid_mask"],inputs["point_conditions"],inputs["global_conditions"],weights.to(device),walkable.to(device),hazard.to(device),codec=codec);terms["loss"].backward();gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),config.gradient_clip);optimizer.step()
        with torch.no_grad():
            for name,value in model.state_dict().items():ema[name].mul_(config.ema_decay).add_(value,alpha=1-config.ema_decay) if value.is_floating_point() else ema[name].copy_(value)
        history.append({"step":step+1,"loss":float(terms["loss"].detach()),"token":float(terms["token"].detach()),"condition":float(terms["condition"].detach()),"point":float(terms["point"].detach()),"reachability":float(terms["reachability"].detach()),"sharpness":float(terms["sharpness"].detach()),"destination_coverage":float(terms["destination_coverage"].detach()),"predicted_hazard":float(terms["predicted_hazard"].detach()),"gradient_norm":float(gradient)})
    model_cpu={name:value.detach().cpu() for name,value in model.state_dict().items()};ema_cpu={name:value.detach().cpu() for name,value in ema.items()};payload={"format":FORMAT,"source_sha256":_source_sha256(),"base_checkpoint_sha256":sha256_file(base_path),"config":config.to_dict(),"base_step":base_step,"semantic_step":semantic_start+updates,"model_state":model_cpu,"ema_state":ema_cpu,"optimizer_state":optimizer.state_dict(),"generator_state":generator.get_state(),"history":history,"model_sha256":tensor_state_sha256(model_cpu),"ema_sha256":tensor_state_sha256(ema_cpu),"token_weights":weights,"walkable_table":walkable,"hazard_table":hazard}
    staging=output.parent/f".{output.name}.tmp-{os.getpid()}-{time.time_ns()}";staging.mkdir(parents=True);torch.save(payload,staging/"checkpoint.pt");report={"format":FORMAT,"status":"passed","base_step":base_step,"semantic_step":semantic_start+updates,"updates":updates,"elapsed_seconds":time.perf_counter()-started,"checkpoint_sha256":sha256_file(staging/"checkpoint.pt"),"model_sha256":payload["model_sha256"],"ema_sha256":payload["ema_sha256"],"last":history[-1]};(staging/"report.json").write_bytes(canonical_json_bytes(report));load_checkpoint(staging/"checkpoint.pt");os.replace(staging,output);return report
