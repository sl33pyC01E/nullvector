from __future__ import annotations
from dataclasses import asdict,dataclass
import hashlib,json
from pathlib import Path
from ..config import PROJECT_ROOT

FORMAT="nullvector-recurrent-action-dit-v2/1.0.0"
CHECKPOINT_FORMAT="nullvector-recurrent-action-dit-v2-checkpoint/1.0.0"
REPORT_FORMAT="nullvector-recurrent-action-dit-v2-report/1.0.0"
BASE_CHECKPOINT=PROJECT_ROOT/"outputs/world_latent_dit/production_v2_residual/checkpoint.pt"
BASE_SHA256="079f28ffa11187167ac7d604132f0b7d37ebd2795c5d03479d8b170ffc57e84a"
CORPUS=PROJECT_ROOT/"outputs/world_action_cellular_v7/corpus_v1_6world"
DEFAULT_OUTPUT=PROJECT_ROOT/"outputs/recurrent_action_dit_v2/production_v1"
SOURCE_FILES=("forge/recurrent_action_dit_v2/__init__.py","forge/recurrent_action_dit_v2/__main__.py","forge/recurrent_action_dit_v2/contract.py","forge/recurrent_action_dit_v2/model.py","forge/recurrent_action_dit_v2/runtime.py","forge/recurrent_action_dit_v2/training.py","forge/world_latent_dit/model.py")

@dataclass(frozen=True,slots=True)
class TrainingPlan:
    total_updates:int=1000
    segment_updates:int=250
    batch_size:int=6
    learning_rate:float=3e-5
    ema_decay:float=.999
    changed_weight:float=6.0
    static_weight:float=.18
    contrastive_weight:float=.12
    seed:int=0x524543555252454E
    def to_dict(self):return asdict(self)

def canonical(value):return(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
def file_sha256(path:Path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1<<20),b""):digest.update(chunk)
    return digest.hexdigest()
def state_sha256(state):
    digest=hashlib.sha256()
    for name,value in sorted(state.items()):digest.update(name.encode()+b"\0"+value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
def source_sha256():
    digest=hashlib.sha256(b"nullvector-recurrent-action-dit-v2\0")
    for relative in SOURCE_FILES:digest.update(relative.encode()+b"\0"+(PROJECT_ROOT/relative).read_bytes()+b"\0")
    return digest.hexdigest()
