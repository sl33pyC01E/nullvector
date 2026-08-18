from __future__ import annotations
from dataclasses import asdict,dataclass
import hashlib,json
from ..config import PROJECT_ROOT

FORMAT="nullvector-whole-viewport-raster-vae/1.0.0"
DEFAULT_CORPUS=PROJECT_ROOT/"outputs/action_teacher_viewport_v5/macro_corpus_v1"
DEFAULT_OUTPUT=PROJECT_ROOT/"outputs/whole_viewport_raster_vae_v1/production_macro_v1"
SOURCE_FILES=("forge/whole_viewport_raster_vae_v1/contract.py","forge/whole_viewport_raster_vae_v1/training.py","forge/world_frame_vae/model.py")

@dataclass(frozen=True)
class TrainingPlan:
    updates:int=2400
    segment_updates:int=200
    batch_size:int=4
    learning_rate:float=2e-5
    ema_decay:float=.999
    seed:int=0x46554C4C56494557

def canonical(value):return (json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
def source_sha256():
    digest=hashlib.sha256(b"nullvector-whole-viewport-raster-vae-v1\0")
    for relative in SOURCE_FILES:digest.update(relative.encode()+b"\0"+(PROJECT_ROOT/relative).read_bytes()+b"\0")
    return digest.hexdigest()
def plan_dict(value):return asdict(value)
