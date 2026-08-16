from __future__ import annotations

from dataclasses import asdict,dataclass
import hashlib
from pathlib import Path

from ..config import PROJECT_ROOT


FORMAT="nullvector-neural-locomotion-2.5d/1.0.0"
CHECKPOINT_FORMAT="nullvector-neural-locomotion-2.5d-checkpoint/1.0.0"
MAX_APPENDAGES=8
MAX_MUSCLES=64
APPENDAGE_FEATURES=16
MUSCLE_FEATURES=8
DYNAMIC_FEATURES=2+2+2+MAX_APPENDAGES
SOURCE_FILES=(
    "forge/creature_stage_neural_locomotion_25d/contract.py",
    "forge/creature_stage_neural_locomotion_25d/data.py",
    "forge/creature_stage_neural_locomotion_25d/model.py",
    "forge/creature_stage_neural_locomotion_25d/training.py",
    "forge/creature_stage_neural_locomotion_25d/evaluation.py",
)


@dataclass(frozen=True,slots=True)
class ModelConfig:
    width:int=384
    recurrent_layers:int=3
    dropout:float=.05

    def __post_init__(self)->None:
        if not 192<=self.width<=768 or not 2<=self.recurrent_layers<=6 or not 0<=self.dropout<=.2:raise ValueError("2.5D neural model config drifted")


@dataclass(frozen=True,slots=True)
class TrainingConfig:
    updates:int=2400
    batch_size:int=24
    learning_rate:float=2e-4
    weight_decay:float=2e-5
    ema_decay:float=.999
    seed:int=0x25D25D

    def __post_init__(self)->None:
        if not 100<=self.updates<=100_000 or not 4<=self.batch_size<=128 or not 1e-6<=self.learning_rate<=1e-3 or not .9<=self.ema_decay<1:raise ValueError("2.5D neural training config drifted")


def source_sha256()->str:
    digest=hashlib.sha256(b"nullvector-neural-locomotion-25d-source-v1\0")
    for relative in SOURCE_FILES:
        path=PROJECT_ROOT/relative
        if not path.is_file():raise FileNotFoundError(relative)
        digest.update(relative.encode()+b"\0"+path.read_bytes()+b"\0")
    return digest.hexdigest()

