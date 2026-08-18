from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib, json
from ..config import PROJECT_ROOT

FORMAT="nullvector-whole-viewport-latent/1.0.0"
CHECKPOINT_FORMAT="nullvector-whole-viewport-latent-checkpoint/1.0.0"
DEFAULT_CORPUS=PROJECT_ROOT/"outputs/action_teacher_viewport_v5/pilot_v1"
DEFAULT_OUTPUT=PROJECT_ROOT/"outputs/whole_viewport_latent_v1/production_v1"
SOURCE_FILES=("forge/whole_viewport_latent_v1/contract.py","forge/whole_viewport_latent_v1/model.py","forge/whole_viewport_latent_v1/data.py","forge/whole_viewport_latent_v1/decoder.py","forge/whole_viewport_latent_v1/training.py","forge/whole_viewport_latent_v1/showcase.py")

@dataclass(frozen=True)
class ModelConfig:
    width:int=192
    blocks:int=12
    latent_channels:int=48
    action_count:int=22
    organism_features:int=164
    spatial_channels:int=68
    actor_field_channels:int=8
    global_features:int=196

@dataclass(frozen=True)
class TrainingConfig:
    steps:int=4000
    batch_size:int=8
    rollout_steps:int=4
    learning_rate:float=2e-4
    weight_decay:float=1e-3
    ema_decay:float=.999
    rgb_weight:float=.15
    stable_rgb_weight:float=4.0
    validation_every:int=250
    checkpoint_every:int=250
    seed:int=0x57484F4C45564945

def canonical(value): return (json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
def source_sha256():
    digest=hashlib.sha256(b"nullvector-whole-viewport-latent-v1\0")
    for relative in SOURCE_FILES:digest.update(relative.encode()+b"\0"+(PROJECT_ROOT/relative).read_bytes()+b"\0")
    return digest.hexdigest()
def config_dict(value): return asdict(value)
