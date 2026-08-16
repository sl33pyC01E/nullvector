from __future__ import annotations
from dataclasses import asdict,dataclass
import hashlib,json
from ..config import PROJECT_ROOT

CHECKPOINT_FORMAT="nullvector-world-frame-vae-pixel-refiner-checkpoint/1.0.0"
SOURCE_FILES=(
    "forge/world_frame_vae/contract.py",
    "forge/world_frame_vae/model.py",
    "forge/world_frame_vae/runtime.py",
    "forge/world_frame_vae_refiner/contract.py",
    "forge/world_frame_vae_refiner/model.py",
    "forge/world_frame_vae_refiner/runtime.py",
    "forge/world_frame_vae_refiner/training.py",
)
@dataclass(frozen=True,slots=True)
class ModelConfig:
    width:int=64
    blocks:int=8
    maximum_delta:float=.24
@dataclass(frozen=True,slots=True)
class TrainingConfig:
    steps:int=5000
    batch_size:int=16
    crop:int=128
    learning_rate:float=2e-4
    ema_decay:float=.999
    seed:int=0x504958454C43454C
def canonical(value):return (json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
def config_dict(value):return asdict(value)
def source_sha256():
    digest=hashlib.sha256(b"nullvector-world-frame-vae-pixel-refiner-v1\0")
    for relative in SOURCE_FILES:digest.update(relative.encode()+b"\0"+(PROJECT_ROOT/relative).read_bytes()+b"\0")
    return digest.hexdigest()
