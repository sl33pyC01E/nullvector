from __future__ import annotations

import hashlib,json
from dataclasses import asdict,dataclass
from pathlib import Path

from ..config import PROJECT_ROOT


FORMAT="nullvector-continuous-organism-cell-vae-v1/1.0.0";CACHE_FORMAT=f"{FORMAT}-cache";CHECKPOINT_FORMAT=f"{FORMAT}-checkpoint";EVALUATION_FORMAT=f"{FORMAT}-evaluation"
DEFAULT_OUTPUT=PROJECT_ROOT/"outputs/organism_cell_vae_v1/production_v1";MAX_CELLS=576;CELL_FEATURES=52
SOURCE_FILES=("forge/organism_cell_vae_v1/__init__.py","forge/organism_cell_vae_v1/__main__.py","forge/organism_cell_vae_v1/contract.py","forge/organism_cell_vae_v1/cache.py","forge/organism_cell_vae_v1/model.py","forge/organism_cell_vae_v1/training.py","forge/organism_cell_vae_v1/evaluation.py")


@dataclass(frozen=True,slots=True)
class Plan:
    total_steps:int=1200;segment_steps:int=200;batch_size:int=4;learning_rate:float=2e-4;ema_decay:float=.995;seed:int=0x43454C4C56414531
    def __post_init__(self)->None:
        if self.total_steps%self.segment_steps or not 100<=self.segment_steps<=self.total_steps<=5000:raise ValueError("cell VAE schedule drifted")
        if not 2<=self.batch_size<=12 or not 0<self.learning_rate<=5e-4 or not .9<=self.ema_decay<1:raise ValueError("cell VAE optimizer drifted")
    def to_dict(self):return asdict(self)


def canonical(value:object)->bytes:return(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
def sha256_file(path:Path)->str:
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1<<20),b""):digest.update(chunk)
    return digest.hexdigest()
def source_sha256()->str:
    digest=hashlib.sha256(b"nullvector-continuous-organism-cell-vae-v1\0")
    for relative in SOURCE_FILES:digest.update(relative.encode()+b"\0"+(PROJECT_ROOT/relative).read_bytes()+b"\0")
    return digest.hexdigest()
