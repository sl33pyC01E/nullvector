from __future__ import annotations

from collections import deque
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from ..config import PROJECT_ROOT
from ..map_topology_neural.compiler import compile_topology, make_raw_topology
from ..map_topology_neural_prior_generation.contract import CODEC_CHECKPOINT_RELATIVE
from ..map_topology_neural_prior_generation.render import case_preview_png_bytes, contact_sheet_png_bytes
from ..map_topology_neural_prior_training.contract import FROZEN_LATENT_CORPUS_RELATIVE
from ..map_topology_neural_prior_training.dataset import PriorTrainingDataset
from ..map_topology_neural_prior_v2.model import build_prior_v2, sample_parallel_v2
from ..map_topology_neural_prior_v2_training.checkpoint import load_checkpoint as load_prior_checkpoint
from ..map_topology_neural_prior_v2_training.contract import PriorV2CalibrationConfig, canonical_json_bytes, sha256_file
from ..map_topology_neural_production.checkpoint import load_checkpoint as load_codec_checkpoint
from ..map_topology_neural.codec import build_codec
from ..map_topology_neural_production.dataset import TopologyProductionDataset
from ..maps.model import WALKABLE_TERRAIN
from ..safety import require_disk_floor


def _reachable(mask: np.ndarray, start: tuple[int, int], targets: tuple[tuple[int, int], ...]) -> bool:
    height, width = mask.shape; sx, sy = start
    if not (0 <= sx < width and 0 <= sy < height and mask[sy, sx]): return False
    seen={(sx,sy)}; queue=deque(((sx,sy),))
    while queue:
        x,y=queue.popleft()
        for point in ((x,y-1),(x-1,y),(x+1,y),(x,y+1)):
            px,py=point
            if 0<=px<width and 0<=py<height and mask[py,px] and point not in seen:seen.add(point);queue.append(point)
    return all(point in seen for point in targets)


def _radius_one(mask: np.ndarray) -> np.ndarray:
    padded=np.pad(mask.astype(bool),1); result=np.ones_like(mask,dtype=bool)
    for dy in range(3):
        for dx in range(3):result&=padded[dy:dy+mask.shape[0],dx:dx+mask.shape[1]]
    return result


def _decode(codec: torch.nn.Module, tokens: torch.Tensor, height: int, width: int):
    table=codec.quantizer.embeddings; embedded=table.index_select(0,tokens.flatten()).view(1,*tokens.shape[-2:],table.shape[1]).permute(0,3,1,2).contiguous()
    with torch.inference_mode(): logits=codec.decode(embedded)
    return tuple(np.ascontiguousarray(logits[name].argmax(1)[0,:height,:width].cpu().numpy().astype(dtype)) for name,dtype in (("terrain",np.uint8),("hazard",np.uint8),("elevation",np.int8)))


def evaluate_checkpoint(checkpoint: Path, output: Path, *, device_name: str="cuda") -> dict[str, object]:
    output=Path(output).resolve()
    if output.exists():raise FileExistsError("Topology-prior evaluation output is immutable.")
    require_disk_floor(output.parent,floor_gb=100.0,planned_bytes=256*1024*1024)
    device=torch.device("cuda" if device_name=="cuda" and torch.cuda.is_available() else "cpu")
    prior_payload=load_prior_checkpoint(Path(checkpoint)); config=PriorV2CalibrationConfig.from_dict(prior_payload["config"])
    prior=build_prior_v2(config.model_config()).to(device);prior.load_state_dict(prior_payload["ema_state"],strict=True);prior.eval()
    codec_payload=load_codec_checkpoint(PROJECT_ROOT/CODEC_CHECKPOINT_RELATIVE);codec_config=codec_payload["config"]
    from ..map_topology_neural_production.contract import TopologyCodecCalibrationConfig
    codec_training=TopologyCodecCalibrationConfig.from_dict(codec_config);codec=build_codec(codec_training.codec_config(),init_seed=codec_training.seed);codec.load_state_dict(codec_payload["ema_state"],strict=True);codec.eval()
    latent=PriorTrainingDataset(PROJECT_ROOT/"outputs/map_decorator_corpus_v1",PROJECT_ROOT/FROZEN_LATENT_CORPUS_RELATIVE)
    source_dataset=TopologyProductionDataset(PROJECT_ROOT/"outputs/map_decorator_corpus_v1");refs=source_dataset.evaluation_refs("test",6);latent_by_id={ref.full_map_identity_sha256:ref for ref in latent.refs_by_split["test"]}
    staging=output.parent/f".{output.name}.tmp-{os.getpid()}-{time.time_ns()}";staging.mkdir(parents=True);rows=[];records=[]
    try:
        for source_ref in refs:
            source=source_dataset.corpus.read_sample(source_ref.shard_id,source_ref.sample_index,expected_split="test");batch=latent.collate((latent_by_id[source_ref.full_map_identity_sha256],));conditions={name:batch[name].to(device) for name in ("valid_mask","point_conditions","global_conditions","theme_index")}
            with torch.inference_mode(),torch.autocast(device_type="cuda",dtype=torch.bfloat16,enabled=device.type=="cuda"):sample=sample_parallel_v2(prior,conditions,sampling_steps=config.sampling_steps)
            terrain,hazard,elevation=_decode(codec,sample["tokens"].cpu(),source.config.height,source.config.width);raw=make_raw_topology(terrain,hazard,elevation,shape=(source.config.height,source.config.width));compiled=compile_topology(raw,seed=config.seed,theme=source.theme,config=source.config,start=source.start,exit=source.exit,objectives=source.objectives,spawns=source.spawns)
            walk=np.isin(terrain,tuple(WALKABLE_TERRAIN));targets=(source.exit,*source.objectives);repair=float(compiled.report["costs"]["repair_fraction"]);name=f"{source.theme}.png";preview=case_preview_png_bytes(source,raw,compiled.data,scale=4);(staging/name).write_bytes(preview);rows.append((source.theme,preview));records.append({"theme":source.theme,"raw_required_reachable":_reachable(walk,source.start,targets),"raw_radius_one_required_reachable":_reachable(_radius_one(walk),source.start,targets),"repair_fraction":repair,"raw_openness":float(walk.mean()),"raw_hazard_fraction":float((hazard!=0).mean()),"unique_tokens":int(torch.unique(sample["tokens"]).numel()),"preview":name})
        contact=contact_sheet_png_bytes(rows);(staging/"contact_sheet.png").write_bytes(contact);report={"format":"nullvector-map-topology-prior-v2-decoded-evaluation/1.0.0","checkpoint_sha256":sha256_file(Path(checkpoint)),"step":prior_payload["step"],"cases":records,"aggregate":{"raw_required_reachable_rate":sum(row["raw_required_reachable"] for row in records)/len(records),"raw_radius_one_required_reachable_rate":sum(row["raw_radius_one_required_reachable"] for row in records)/len(records),"mean_repair_fraction":sum(row["repair_fraction"] for row in records)/len(records),"mean_unique_tokens":sum(row["unique_tokens"] for row in records)/len(records)},"contact_sheet_sha256":hashlib.sha256(contact).hexdigest()};report["report_sha256"]=hashlib.sha256(canonical_json_bytes(report)).hexdigest();(staging/"evaluation.json").write_bytes(canonical_json_bytes(report));os.replace(staging,output);return report
    finally:
        if staging.exists():
            for child in staging.iterdir():child.unlink(missing_ok=True)
            staging.rmdir()
