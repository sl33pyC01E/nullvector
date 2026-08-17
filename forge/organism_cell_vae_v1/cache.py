from __future__ import annotations

import hashlib,json,math,os
from pathlib import Path
import tempfile

import numpy as np
import torch

from ..creature_stage_developmental.contract import APPENDAGE_KINDS,TISSUES
from ..organism_raster_vae_v5_anatomical.dataset import AnatomicalGraphCorpus,_raster_positions
from ..safety import require_disk_floor
from .contract import CACHE_FORMAT,CELL_FEATURES,MAX_CELLS,canonical,sha256_file,source_sha256


CACHE_NAME="cell_field_cache.pt";MANIFEST_NAME="cell_field_cache_manifest.json"


def _atomic(path:Path,value:bytes)->None:
    descriptor,name=tempfile.mkstemp(prefix=f".{path.name}.tmp-",dir=path.parent);temporary=Path(name)
    try:
        with os.fdopen(descriptor,"wb") as handle:handle.write(value);handle.flush();os.fsync(handle.fileno())
        os.replace(temporary,path)
    finally:temporary.unlink(missing_ok=True)


def build(root:Path)->dict[str,object]:
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    if (root/CACHE_NAME).exists() or (root/MANIFEST_NAME).exists():return validate(root)
    require_disk_floor(root,floor_gb=100,planned_bytes=256*1024**2);corpus=AnatomicalGraphCorpus();count=len(corpus)
    features=torch.zeros(count,MAX_CELLS,CELL_FEATURES,dtype=torch.float16);mask=torch.zeros(count,MAX_CELLS,dtype=torch.bool);targets=torch.empty(count,4,96,96,dtype=torch.float16);identities=torch.empty(count,dtype=torch.int16);phases=torch.empty(count,dtype=torch.uint8)
    for index in range(count):
        row=corpus[index];identity,phase_index=corpus.rows[index];organism=corpus.organisms[identity];cells,nodes=_raster_positions(organism,phase_index);n=organism.cell_count
        if n>MAX_CELLS:raise ValueError("continuous cell census exceeded")
        family=int(np.argmax(organism.genome.family_mix));value=np.zeros((n,CELL_FEATURES),np.float32);value[:,:2]=cells/47*2-1;value[:,2:17]=np.eye(len(TISSUES),dtype=np.float32)[organism.tissue];value[:,17+family]=1;value[:,22:37]=organism.trait_fields
        appendage=np.zeros((n,9),np.float32);appendage[:,0]=organism.appendage_index<0
        for cell,appendage_index in enumerate(organism.appendage_index):
            if appendage_index>=0:appendage[cell,1+APPENDAGE_KINDS.index(organism.genome.appendages[int(appendage_index)].kind)]=1
        value[:,37:46]=appendage;value[:,46]=organism.side;phase=phase_index/16;value[:,47]=math.sin(math.tau*phase);value[:,48]=math.cos(math.tau*phase);value[:,49]=organism.component_weights.max(1);value[:,50]=organism.appendage_index>=0;value[:,51]=1
        features[index,:n]=torch.from_numpy(value).half();mask[index,:n]=True;targets[index]=row["rgba"].half();identities[index]=identity;phases[index]=phase_index
    payload={"format":CACHE_FORMAT,"source_sha256":source_sha256(),"corpus_sha256":corpus.semantic_sha256,"features":features,"mask":mask,"target_rgba":targets,"identity":identities,"phase_index":phases}
    temporary=root/f".{CACHE_NAME}.tmp-{os.getpid()}";torch.save(payload,temporary);os.replace(temporary,root/CACHE_NAME)
    manifest={"format":CACHE_FORMAT,"source_sha256":source_sha256(),"corpus_sha256":corpus.semantic_sha256,"count":count,"max_cells":MAX_CELLS,"feature_count":CELL_FEATURES,"cache":{"path":CACHE_NAME,"bytes":(root/CACHE_NAME).stat().st_size,"sha256":sha256_file(root/CACHE_NAME)}};manifest["manifest_sha256"]=hashlib.sha256(canonical(manifest)).hexdigest();_atomic(root/MANIFEST_NAME,canonical(manifest));return validate(root)


def validate(root:Path)->dict[str,object]:
    root=Path(root).resolve();raw=(root/MANIFEST_NAME).read_bytes();manifest=json.loads(raw)
    if raw!=canonical(manifest) or manifest.get("format")!=CACHE_FORMAT or manifest.get("source_sha256")!=source_sha256():raise ValueError("cell VAE cache manifest drifted")
    expected=hashlib.sha256(canonical({k:v for k,v in manifest.items() if k!="manifest_sha256"})).hexdigest()
    if manifest.get("manifest_sha256")!=expected:raise ValueError("cell VAE cache hash drifted")
    path=root/manifest["cache"]["path"]
    if path.stat().st_size!=manifest["cache"]["bytes"] or sha256_file(path)!=manifest["cache"]["sha256"]:raise ValueError("cell VAE cache artifact drifted")
    return manifest


def load(root:Path):
    validate(root);payload=torch.load(Path(root)/CACHE_NAME,map_location="cpu",weights_only=True);required={"format","source_sha256","corpus_sha256","features","mask","target_rgba","identity","phase_index"}
    if set(payload)!=required or payload["format"]!=CACHE_FORMAT or payload["source_sha256"]!=source_sha256():raise ValueError("cell VAE cache payload drifted")
    count=len(payload["identity"])
    if payload["features"].shape!=(count,MAX_CELLS,CELL_FEATURES) or payload["mask"].shape!=(count,MAX_CELLS) or payload["target_rgba"].shape!=(count,4,96,96):raise ValueError("cell VAE cache geometry drifted")
    return payload
