from __future__ import annotations
import hashlib,json
from ..config import PROJECT_ROOT

FORMAT="nullvector-action-teacher-trajectory/1.1.0";FRAME_SIZE=(256,256);STATE_FEATURES=64;COUNTERFACTUAL_SHAPE=(5,4)
ACTIONS=("none","inspect","impact","heal","scrape","cut","beam","projectile","interact","build","craft","bond","graft_organ","graft_locomotor","ability_up","ability_right","ability_down","ability_left","intervention","trade","service","metamorphosis")
SOURCE_FILES=("forge/action_teacher_v1/contract.py","forge/action_teacher_v1/recorder.py")
def canonical(value):return (json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
def source_sha256():
    digest=hashlib.sha256(b"nullvector-action-teacher-v1\0")
    for relative in SOURCE_FILES:digest.update(relative.encode()+b"\0"+(PROJECT_ROOT/relative).read_bytes()+b"\0")
    return digest.hexdigest()
