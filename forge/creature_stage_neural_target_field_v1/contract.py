from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from ..config import PROJECT_ROOT

FORMAT="nullvector-neural-grounded-target-field/1.0.0"
CHECKPOINT_FORMAT="nullvector-neural-grounded-target-field-checkpoint/1.0.0"
TARGET_PHASE_HARMONICS=8
TARGET_FEATURES=4+TARGET_PHASE_HARMONICS*2
TARGET_OWNER_FEATURES=16
TARGET_GLOBAL_FEATURES=20
SOURCE_FILES=(
 "forge/creature_stage_neural_target_field_v1/contract.py",
 "forge/creature_stage_neural_target_field_v1/dataset.py",
 "forge/creature_stage_neural_target_field_v1/model.py",
 "forge/creature_stage_neural_target_field_v1/runtime.py",
 "forge/creature_stage_neural_target_field_v1/physics.py",
 "forge/creature_stage_neural_target_field_v1/training.py",
 "forge/creature_stage_neural_grounded_feedback_v2/dataset.py",
 "forge/creature_stage_neural_grounded_feedback_v2/model.py",
 "forge/creature_stage_neural_grounded_feedback_v2/physics.py",
 "forge/creature_stage_neural_grounded_cyclic/curriculum.py",
 "forge/creature_stage_developmental/motion.py",
)

@dataclass(frozen=True,slots=True)
class ModelConfig:
    width:int=256; depth:int=4; dropout:float=.02
    def to_dict(self): return asdict(self)

@dataclass(frozen=True,slots=True)
class TrainingConfig:
    updates:int=5000; batch_size:int=256; learning_rate:float=2e-4; ema_decay:float=.997; variants_per_family:int=4; target_variants_per_chassis:int=3; seed:int=0x5441524745544631
    def to_dict(self): return asdict(self)

def source_sha256()->str:
    d=hashlib.sha256(b"nullvector-neural-grounded-target-field-source-v1\0")
    for rel in SOURCE_FILES:
        p=PROJECT_ROOT/rel
        if not p.is_file(): raise FileNotFoundError(rel)
        d.update(rel.encode()+b"\0"+p.read_bytes()+b"\0")
    return d.hexdigest()
