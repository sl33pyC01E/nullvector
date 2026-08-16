from __future__ import annotations

from dataclasses import asdict,dataclass
import hashlib,json
from pathlib import Path

from ..config import PROJECT_ROOT

FORMAT="nullvector-neural-colony-coordinator/1.0.0"
CHECKPOINT_FORMAT="nullvector-neural-colony-coordinator-checkpoint/1.0.0"
FEATURES=64
MAX_MEMBERS=16
ROLES=("gatherer","scout","defender","medic","breeder","builder")
SOURCE_FILES=("forge/nature_colony_nn/contract.py","forge/nature_colony_nn/corpus.py","forge/nature_colony_nn/model.py","forge/nature_colony_nn/training.py","forge/nature_colony_nn/runtime.py","forge/nature_sim_v2/phenotype.py","forge/nature_sim_v2/colony_ecology.py")

@dataclass(frozen=True,slots=True)
class ModelConfig:
    width:int=384
    layers:int=6
    heads:int=8
    dropout:float=.06

@dataclass(frozen=True,slots=True)
class TrainingConfig:
    steps:int=1200
    batch_size:int=64
    learning_rate:float=2e-4
    seed:int=0x434F4C4F4E59
    ema_decay:float=.999

def canonical(value)->bytes:return (json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
def source_sha256()->str:
    digest=hashlib.sha256(b"nullvector-colony-nn-v1\0")
    for relative in SOURCE_FILES:
        path=PROJECT_ROOT/relative;digest.update(relative.encode()+b"\0"+path.read_bytes()+b"\0")
    return digest.hexdigest()
def config_dict(value):return asdict(value)
