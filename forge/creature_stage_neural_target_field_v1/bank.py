from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

import numpy as np
import torch

from ..config import PROJECT_ROOT
from ..creature_stage_neural_grounded_feedback_v2.dataset import FeedbackCorpus
from ..multifield_style_motion.hashing import deterministic_npz_bytes
from ..safety import require_disk_floor
from .dataset import (
    TargetAugmentationCorpus, TargetFieldCorpus, build_target_augmentation,
    build_target_corpus,
)

BANK_FORMAT="nullvector-neural-grounded-target-training-bank/1.0.0"
BANK_SOURCE_FILES=(
    "forge/creature_stage_neural_target_field_v1/contract.py",
    "forge/creature_stage_neural_target_field_v1/dataset.py",
    "forge/creature_stage_neural_target_field_v1/bank.py",
    "forge/creature_stage_neural_grounded_feedback_v2/contract.py",
    "forge/creature_stage_neural_grounded_feedback_v2/dataset.py",
    "forge/creature_stage_neural_grounded_cyclic/curriculum.py",
    "forge/creature_stage_developmental/contract.py",
    "forge/creature_stage_developmental/development.py",
    "forge/creature_stage_developmental/genomes.py",
    "forge/creature_stage_developmental/motion.py",
)
TRAIN_FEEDBACK_FIELDS=(
    "owner_state","global_state","owner_mask","muscle_meta","muscle_owner",
    "muscle_mask","muscle_target","contact_target","body_target","identity","frame",
)
TRAIN_FIELDS=TRAIN_FEEDBACK_FIELDS+("target_context","terminal_target")
AUG_FIELDS=(
    "owner_state","global_state","owner_mask","muscle_meta","muscle_owner",
    "muscle_mask","target_context","terminal_target","family",
)

def bank_source_sha256()->str:
    digest=hashlib.sha256(b"nullvector-target-training-bank-source-v1\0")
    for relative in BANK_SOURCE_FILES:
        path=PROJECT_ROOT/relative
        if not path.is_file(): raise FileNotFoundError(relative)
        digest.update(relative.encode()+b"\0"+path.read_bytes()+b"\0")
    return digest.hexdigest()

def _canonical(value)->bytes:
    return (json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=True)+"\n").encode()

def _semantic(arrays:dict[str,np.ndarray])->str:
    digest=hashlib.sha256(b"nullvector-target-training-bank-arrays-v1\0")
    for name in sorted(arrays):
        value=np.ascontiguousarray(arrays[name]);digest.update(name.encode()+b"\0")
        digest.update(value.dtype.str.encode()+b"\0"+np.asarray(value.shape,dtype="<i8").tobytes()+memoryview(value))
    return digest.hexdigest()

def build_training_bank(output:Path,*,variants_per_family:int=4,target_variants_per_chassis:int=3)->dict:
    output=Path(output).resolve();require_disk_floor(output.parent,floor_gb=100,planned_bytes=1024**3)
    if output.exists(): raise FileExistsError(output)
    train=build_target_corpus(split="train",variants_per_family=variants_per_family)
    aug=build_target_augmentation(variants_per_chassis=target_variants_per_chassis)
    arrays={f"train_{name}":np.ascontiguousarray(getattr(train.feedback,name).numpy()) for name in TRAIN_FEEDBACK_FIELDS}
    arrays.update({"train_target_context":np.ascontiguousarray(train.target_context.numpy()),"train_terminal_target":np.ascontiguousarray(train.terminal_target.numpy())})
    arrays.update({f"aug_{name}":np.ascontiguousarray(getattr(aug,name).numpy()) for name in AUG_FIELDS})
    archive=deterministic_npz_bytes(arrays);stage=output.parent/f".{output.name}.tmp-{uuid.uuid4().hex}";stage.mkdir(parents=True)
    path=stage/"training_arrays.npz";path.write_bytes(archive)
    manifest={"format":BANK_FORMAT,"source_sha256":bank_source_sha256(),"variants_per_family":variants_per_family,"target_variants_per_chassis":target_variants_per_chassis,
        "train_samples":train.samples,"augmentation_samples":aug.samples,"train_semantic_sha256":train.semantic_sha256,"augmentation_semantic_sha256":aug.semantic_sha256,
        "arrays":{"path":path.name,"bytes":len(archive),"sha256":hashlib.sha256(archive).hexdigest(),"semantic_sha256":_semantic(arrays),"members":sorted(arrays)}}
    manifest["semantic_sha256"]=hashlib.sha256(_canonical(manifest)).hexdigest();(stage/"bank_manifest.json").write_bytes(_canonical(manifest));os.replace(stage,output)
    validate_training_bank(output);return manifest

def validate_training_bank(root:Path,*,require_current_source:bool=True)->dict:
    root=Path(root).resolve();raw=(root/"bank_manifest.json").read_bytes();manifest=json.loads(raw)
    if raw!=_canonical(manifest): raise ValueError("target training bank manifest is not canonical")
    semantic=manifest.pop("semantic_sha256")
    if semantic!=hashlib.sha256(_canonical(manifest)).hexdigest(): raise ValueError("target training bank semantic hash drifted")
    manifest["semantic_sha256"]=semantic
    if manifest.get("format")!=BANK_FORMAT or (require_current_source and manifest.get("source_sha256")!=bank_source_sha256()): raise ValueError("target training bank provenance drifted")
    artifact=manifest["arrays"];path=root/artifact["path"]
    if not path.is_file() or path.stat().st_size!=artifact["bytes"] or path.stat().st_size>256*1024**2: raise ValueError("target training bank size drifted")
    if hashlib.sha256(path.read_bytes()).hexdigest()!=artifact["sha256"]: raise ValueError("target training bank artifact drifted")
    with np.load(path,allow_pickle=False,max_header_size=64*1024) as archive:
        if sorted(archive.files)!=artifact["members"]: raise ValueError("target training bank member census drifted")
        arrays={name:np.ascontiguousarray(archive[name]) for name in archive.files}
    if _semantic(arrays)!=artifact["semantic_sha256"]: raise ValueError("target training bank array identity drifted")
    if arrays["train_owner_state"].shape[0]!=manifest["train_samples"] or arrays["aug_owner_state"].shape[0]!=manifest["augmentation_samples"]: raise ValueError("target training bank sample census drifted")
    return manifest

def rebind_training_bank(source:Path,output:Path)->dict:
    source=Path(source).resolve();output=Path(output).resolve();require_disk_floor(output.parent,floor_gb=100,planned_bytes=1024**3)
    if output.exists(): raise FileExistsError(output)
    parent=validate_training_bank(source,require_current_source=False);stage=output.parent/f".{output.name}.tmp-{uuid.uuid4().hex}";stage.mkdir(parents=True)
    artifact_name=parent["arrays"]["path"];shutil.copyfile(source/artifact_name,stage/artifact_name)
    manifest={key:value for key,value in parent.items() if key!="semantic_sha256"};manifest["source_sha256"]=bank_source_sha256();manifest["derived_from_semantic_sha256"]=parent["semantic_sha256"]
    manifest["semantic_sha256"]=hashlib.sha256(_canonical(manifest)).hexdigest();(stage/"bank_manifest.json").write_bytes(_canonical(manifest));os.replace(stage,output)
    validate_training_bank(output);return manifest

def load_training_bank(root:Path)->tuple[TargetFieldCorpus,TargetAugmentationCorpus]:
    root=Path(root).resolve();manifest=validate_training_bank(root)
    with np.load(root/manifest["arrays"]["path"],allow_pickle=False,max_header_size=64*1024) as archive:
        arrays={name:np.ascontiguousarray(archive[name]) for name in archive.files}
    feedback=FeedbackCorpus(*(torch.from_numpy(arrays[f"train_{name}"]) for name in TRAIN_FEEDBACK_FIELDS),manifest["train_semantic_sha256"],(),())
    train=TargetFieldCorpus(feedback,torch.from_numpy(arrays["train_target_context"]),torch.from_numpy(arrays["train_terminal_target"]),manifest["train_semantic_sha256"])
    aug=TargetAugmentationCorpus(*(torch.from_numpy(arrays[f"aug_{name}"]) for name in AUG_FIELDS),manifest["augmentation_semantic_sha256"])
    return train,aug

def main()->None:
    import argparse
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest="command",required=True)
    build=sub.add_parser("build");build.add_argument("output",type=Path);build.add_argument("--variants-per-family",type=int,default=4);build.add_argument("--target-variants-per-chassis",type=int,default=3)
    check=sub.add_parser("validate");check.add_argument("root",type=Path)
    rebind=sub.add_parser("rebind");rebind.add_argument("source",type=Path);rebind.add_argument("output",type=Path)
    args=parser.parse_args()
    if args.command=="build": result=build_training_bank(args.output,variants_per_family=args.variants_per_family,target_variants_per_chassis=args.target_variants_per_chassis)
    elif args.command=="rebind": result=rebind_training_bank(args.source,args.output)
    else: result=validate_training_bank(args.root)
    print(json.dumps(result,indent=2,sort_keys=True))

if __name__=="__main__": main()
