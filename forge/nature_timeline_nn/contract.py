from __future__ import annotations
from dataclasses import asdict,dataclass
import hashlib,json
from ..config import PROJECT_ROOT

FORMAT="nullvector-neural-world-timeline/1.0.0";CHECKPOINT_FORMAT="nullvector-neural-world-timeline-checkpoint/1.0.0";FEATURES=64;SEQUENCE=24;EVENTS=("quiet","birth","death","predation","mutation","colony","climate","construction","discovery","migration")
SOURCE_FILES=("forge/nature_timeline_nn/contract.py","forge/nature_timeline_nn/model.py","forge/nature_timeline_nn/corpus.py","forge/nature_timeline_nn/training.py","forge/nature_timeline_nn/runtime.py")
@dataclass(frozen=True,slots=True)
class ModelConfig:width:int=512;layers:int=8;heads:int=8;dropout:float=.05
@dataclass(frozen=True,slots=True)
class TrainingConfig:steps:int=1400;batch_size:int=192;learning_rate:float=2e-4;ema_decay:float=.999;seed:int=0x54494D454C494E45
def canonical(v):return (json.dumps(v,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
def source_sha256():
    d=hashlib.sha256(b"nullvector-world-timeline-v1\0")
    for rel in SOURCE_FILES:d.update(rel.encode()+b"\0"+(PROJECT_ROOT/rel).read_bytes()+b"\0")
    return d.hexdigest()
def config_dict(v):return asdict(v)
