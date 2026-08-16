from __future__ import annotations

from dataclasses import asdict,dataclass
import hashlib,json

from ..config import PROJECT_ROOT


FORMAT="nullvector-neural-society-strategist/1.0.0"
CHECKPOINT_FORMAT="nullvector-neural-society-strategist-checkpoint/1.0.0"
FEATURES=64
LABOR_SECTORS=("forage","construction","medicine","defense","research","trade")
PROJECTS=("habitat","workshop","clinic","granary","observatory","graft_house","battery_hall","shrine","market")
DIPLOMACY=("cooperate","neutral","hostile")
ACTIVITIES=("forage","hunt","heal","breed","graft","craft","build","trade","explore","map","study_anomaly","defend","negotiate","raid","found_colony","recover_relic")
SOURCE_FILES=("forge/nature_society_nn/contract.py","forge/nature_society_nn/corpus.py","forge/nature_society_nn/model.py","forge/nature_society_nn/training.py","forge/nature_society_nn/runtime.py")


@dataclass(frozen=True,slots=True)
class ModelConfig:
    width:int=768
    depth:int=8
    dropout:float=.04


@dataclass(frozen=True,slots=True)
class TrainingConfig:
    steps:int=1600
    batch_size:int=512
    learning_rate:float=2.5e-4
    seed:int=0x534F4349455459
    ema_decay:float=.9985


def canonical(value)->bytes:return (json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
def config_dict(value):return asdict(value)
def source_sha256()->str:
    digest=hashlib.sha256(b"nullvector-neural-society-v1\0")
    for relative in SOURCE_FILES:digest.update(relative.encode()+b"\0"+(PROJECT_ROOT/relative).read_bytes()+b"\0")
    return digest.hexdigest()
