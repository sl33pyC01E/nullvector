from __future__ import annotations
from dataclasses import asdict,dataclass
import hashlib,json
from ..config import PROJECT_ROOT

CHECKPOINT_FORMAT="nullvector-world-latent-residual-action-dit-checkpoint/2.0.0"
SOURCE_FILES=(
    "forge/world_latent_dit/model.py",
    "forge/world_latent_dit/data.py",
    "forge/world_latent_dit_v2/contract.py",
    "forge/world_latent_dit_v2/runtime.py",
    "forge/world_latent_dit_v2/training.py",
)

@dataclass(frozen=True,slots=True)
class ModelConfig:
    width:int=512
    layers:int=8
    heads:int=8
    patch:int=4

@dataclass(frozen=True,slots=True)
class TrainingConfig:
    steps:int=5000
    batch_size:int=16
    learning_rate:float=2e-4
    ema_decay:float=.9995
    horizon:int=4
    changed_weight:float=4.0
    static_weight:float=.20
    input_noise:float=.008
    seed:int=0x524553494455414C

def canonical(value):return (json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
def config_dict(value):return asdict(value)
def source_sha256():
    digest=hashlib.sha256(b"nullvector-world-latent-residual-dit-v2\0")
    for relative in SOURCE_FILES:digest.update(relative.encode()+b"\0"+(PROJECT_ROOT/relative).read_bytes()+b"\0")
    return digest.hexdigest()
