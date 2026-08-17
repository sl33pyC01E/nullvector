from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch

from ..organism_raster_vae_v3.contract import RasterVAEV3Config
from ..organism_raster_vae_v5_anatomical.dataset import AnatomicalGraphCorpus
from ..organism_raster_vae_v5_anatomical.model import AnatomicalGraphRasterVAE
from ..organism_raster_vae_v5_anatomical.training import _batch
from ..organism_raster_vae_v6_current.training import source_sha256 as parent_source_sha256
from ..safety import require_disk_floor
from .contract import CACHE_FORMAT, PARENT_CHECKPOINT, canonical, sha256_file, source_sha256


CACHE_NAME = "refiner_cache.pt"
CACHE_MANIFEST = "refiner_cache_manifest.json"


def _atomic(path: Path, value: bytes) -> None:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent); temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle: handle.write(value); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally: temporary.unlink(missing_ok=True)


@torch.inference_mode()
def build(root: Path, *, device_name: str = "cuda") -> dict[str, object]:
    root = Path(root).resolve(); root.mkdir(parents=True, exist_ok=True)
    cache_path = root / CACHE_NAME; manifest_path = root / CACHE_MANIFEST
    if cache_path.exists() or manifest_path.exists():
        return validate(root)
    require_disk_floor(root, floor_gb=100, planned_bytes=512 * 1024**2)
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=True)
    if parent.get("source_sha256") != parent_source_sha256(): raise ValueError("V7 parent source drifted")
    corpus = AnatomicalGraphCorpus(); device = torch.device(device_name)
    model = AnatomicalGraphRasterVAE(RasterVAEV3Config(**parent["config"])); model.load_state_dict(parent["ema_state"], strict=True); model.to(device).eval()
    living=[]; parent_rgba=[]; target=[]; appendage=[]; identity=[]; phase=[]
    for start in range(0, len(corpus), 8):
        indices=list(range(start,min(start+8,len(corpus)))); batch=_batch(corpus,indices,device)
        output=model(batch["living"],batch["family"],batch["traits"],batch["phase"],batch["tokens"],batch["token_mask"],stochastic=False)
        living.append(batch["living"].half().cpu()); parent_rgba.append(output.rgba.half().cpu()); target.append(batch["rgba"].half().cpu()); appendage.append(batch["appendage_alpha"].half().cpu()); identity.append(batch["identity"].short().cpu()); phase.append(batch["phase_index"].byte().cpu())
    payload={"format":CACHE_FORMAT,"source_sha256":source_sha256(),"parent_checkpoint_sha256":sha256_file(PARENT_CHECKPOINT),"parent_model_state_sha256":parent["ema_state_sha256"],"corpus_sha256":corpus.semantic_sha256,"living":torch.cat(living),"parent_rgba":torch.cat(parent_rgba),"target_rgba":torch.cat(target),"appendage_alpha":torch.cat(appendage),"identity":torch.cat(identity),"phase_index":torch.cat(phase)}
    temporary=root/f".{CACHE_NAME}.tmp-{os.getpid()}"; torch.save(payload,temporary); os.replace(temporary,cache_path)
    manifest={"format":CACHE_FORMAT,"source_sha256":source_sha256(),"corpus_sha256":corpus.semantic_sha256,"count":len(corpus),"cache":{"path":CACHE_NAME,"bytes":cache_path.stat().st_size,"sha256":sha256_file(cache_path)},"parent_checkpoint_sha256":sha256_file(PARENT_CHECKPOINT)}
    manifest["manifest_sha256"]=hashlib.sha256(canonical(manifest)).hexdigest(); _atomic(manifest_path,canonical(manifest)); return validate(root)


def load(root: Path) -> dict[str, object]:
    validate(root); payload=torch.load(Path(root)/CACHE_NAME,map_location="cpu",weights_only=True)
    required={"format","source_sha256","parent_checkpoint_sha256","parent_model_state_sha256","corpus_sha256","living","parent_rgba","target_rgba","appendage_alpha","identity","phase_index"}
    if set(payload)!=required or payload["format"]!=CACHE_FORMAT or payload["source_sha256"]!=source_sha256(): raise ValueError("V7 cache payload drifted")
    count=len(payload["identity"])
    if payload["living"].shape!=(count,42,48,48) or payload["parent_rgba"].shape!=payload["target_rgba"].shape!=(count,4,96,96) or payload["appendage_alpha"].shape!=(count,1,96,96): raise ValueError("V7 cache tensor geometry drifted")
    return payload


def validate(root: Path) -> dict[str, object]:
    root=Path(root).resolve(); raw=(root/CACHE_MANIFEST).read_bytes(); manifest=json.loads(raw)
    if raw!=canonical(manifest) or manifest.get("format")!=CACHE_FORMAT or manifest.get("source_sha256")!=source_sha256(): raise ValueError("V7 cache manifest drifted")
    expected=hashlib.sha256(canonical({k:v for k,v in manifest.items() if k!="manifest_sha256"})).hexdigest()
    if manifest.get("manifest_sha256")!=expected: raise ValueError("V7 cache manifest hash drifted")
    path=root/manifest["cache"]["path"]
    if path.stat().st_size!=manifest["cache"]["bytes"] or sha256_file(path)!=manifest["cache"]["sha256"]: raise ValueError("V7 cache artifact drifted")
    return manifest
