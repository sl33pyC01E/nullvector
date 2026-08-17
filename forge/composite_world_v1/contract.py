from __future__ import annotations
import hashlib,json
from pathlib import Path
from ..config import PROJECT_ROOT
FORMAT="nullvector-composite-neural-world-v1/1.1.0";DEFAULT_OUTPUT=PROJECT_ROOT/"outputs/composite_world_v1/build_002"
SOURCE_FILES=("forge/composite_world_v1/__init__.py","forge/composite_world_v1/__main__.py","forge/composite_world_v1/contract.py","forge/composite_world_v1/runtime.py","forge/composite_world_v1/release.py")
ARTIFACTS={
"teacher_ensemble":("outputs/neural_ensemble_v1/build_001/ensemble_manifest.json",None),
"action_dit":("outputs/world_latent_dit/production_v2_residual/report.json","outputs/world_latent_dit/production_v2_residual/checkpoint.pt"),
"world_vae":("outputs/world_frame_vae/production_v2_high_fidelity/report.json","outputs/world_frame_vae/production_v2_high_fidelity/checkpoint.pt"),
"pixel_refiner":("outputs/world_frame_refiner_v2/production_v1/report.json","outputs/world_frame_refiner_v2/production_v1/refiner_0003000.pt"),
"actor_state":("outputs/actor_state_student_v1/production_v1/report.json","outputs/actor_state_student_v1/production_v1/actor_0000800.pt"),
"organism_vae":("outputs/organism_cell_vae_v1/production_v3_calibrated/evaluation_manifest.json","outputs/organism_cell_vae_v1/production_v3_calibrated/cell_vae_0001200.pt"),
"physiology":("outputs/cellular_nca/nca_causal_v3_selected/selection_manifest.json","outputs/cellular_nca/nca_causal_v3_selected/runtime.pt"),
}
def canonical(value):return(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
def file_sha256(path:Path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1<<20),b""):digest.update(chunk)
    return digest.hexdigest()
def source_sha256():
    digest=hashlib.sha256(b"nullvector-composite-neural-world-v1\0")
    for relative in SOURCE_FILES:digest.update(relative.encode()+b"\0"+(PROJECT_ROOT/relative).read_bytes()+b"\0")
    return digest.hexdigest()
def tensor_state_sha256(state):
    digest=hashlib.sha256()
    for name,value in sorted(state.items()):digest.update(name.encode()+b"\0"+value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
