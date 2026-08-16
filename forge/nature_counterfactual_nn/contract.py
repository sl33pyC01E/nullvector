from __future__ import annotations
from dataclasses import asdict,dataclass
import hashlib,json
from ..config import PROJECT_ROOT

FORMAT="nullvector-neural-ecology-counterfactual/1.0.0";CHECKPOINT_FORMAT="nullvector-neural-ecology-counterfactual-checkpoint/1.0.0";FEATURES=64;SEQUENCE=24
ACTIONS=("seed_ark","climate_condenser","phase_anchor","predator_ward","habitat_knot")
SOURCE_FILES=("forge/nature_counterfactual_nn/contract.py","forge/nature_counterfactual_nn/model.py","forge/nature_counterfactual_nn/corpus.py","forge/nature_counterfactual_nn/training.py","forge/nature_counterfactual_nn/runtime.py")
@dataclass(frozen=True,slots=True)
class ModelConfig:width:int=512;layers:int=8;heads:int=8;dropout:float=.04
@dataclass(frozen=True,slots=True)
class TrainingConfig:steps:int=2800;batch_size:int=96;learning_rate:float=2e-4;ema_decay:float=.9992;seed:int=0x434F554E544552
def canonical(value):return (json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
def source_sha256():
    digest=hashlib.sha256(b"nullvector-counterfactual-v1\0")
    for relative in SOURCE_FILES:digest.update(relative.encode()+b"\0"+(PROJECT_ROOT/relative).read_bytes()+b"\0")
    return digest.hexdigest()
def config_dict(value):return asdict(value)
