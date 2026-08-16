from __future__ import annotations
from dataclasses import asdict,dataclass
import hashlib,json
from ..config import PROJECT_ROOT

FORMAT="nullvector-world-latent-action-dit/1.0.0";CHECKPOINT_FORMAT="nullvector-world-latent-action-dit-checkpoint/1.0.0";LATENT_CHANNELS=48;LATENT_SIZE=32;STATE_FEATURES=64;CONTROL_FEATURES=4;ACTIONS=22
SOURCE_FILES=("forge/world_latent_dit/contract.py","forge/world_latent_dit/model.py","forge/world_latent_dit/data.py","forge/world_latent_dit/training.py","forge/world_latent_dit/runtime.py")
@dataclass(frozen=True,slots=True)
class ModelConfig:width:int=512;layers:int=8;heads:int=8;patch:int=4
@dataclass(frozen=True,slots=True)
class TrainingConfig:steps:int=5000;batch_size:int=32;learning_rate:float=2e-4;ema_decay:float=.9995;horizon:int=4;seed:int=0x414354494F4E4449
def canonical(value):return (json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
def source_sha256():
    digest=hashlib.sha256(b"nullvector-world-action-dit-v1\0")
    for relative in SOURCE_FILES:digest.update(relative.encode()+b"\0"+(PROJECT_ROOT/relative).read_bytes()+b"\0")
    return digest.hexdigest()
def config_dict(value):return asdict(value)
