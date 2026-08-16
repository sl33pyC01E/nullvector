from __future__ import annotations
from dataclasses import asdict,dataclass
import hashlib,json
from ..config import PROJECT_ROOT

FORMAT="nullvector-world-frame-vae/1.0.0";CHECKPOINT_FORMAT="nullvector-world-frame-vae-checkpoint/1.0.0";SIZE=256
SOURCE_FILES=("forge/world_frame_vae/contract.py","forge/world_frame_vae/model.py","forge/world_frame_vae/data.py","forge/world_frame_vae/training.py","forge/world_frame_vae/runtime.py")
@dataclass(frozen=True,slots=True)
class ModelConfig:base:int=128;latent_channels:int=48
@dataclass(frozen=True,slots=True)
class TrainingConfig:steps:int=4800;batch_size:int=8;learning_rate:float=1.8e-4;ema_decay:float=.9995;kl_weight:float=5e-6;seed:int=0x574F524C44564145
def canonical(value):return (json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
def source_sha256():
    digest=hashlib.sha256(b"nullvector-world-frame-vae-v1\0")
    for relative in SOURCE_FILES:digest.update(relative.encode()+b"\0"+(PROJECT_ROOT/relative).read_bytes()+b"\0")
    return digest.hexdigest()
def config_dict(value):return asdict(value)
