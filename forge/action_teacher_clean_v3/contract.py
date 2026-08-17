from __future__ import annotations

import hashlib
import json

from ..action_teacher_v1.contract import ACTIONS,COUNTERFACTUAL_SHAPE,FRAME_SIZE,STATE_FEATURES
from ..action_teacher_v2.contract import ACTOR_FEATURES,ACTOR_FIELD_SHAPE
from ..config import PROJECT_ROOT


FORMAT="nullvector-clean-cellular-action-teacher/3.0.0"
DEFAULT_ROOT=PROJECT_ROOT/"outputs/action_teacher_clean_v3/production_v1"
SOURCE_FILES=(
    "forge/action_teacher_clean_v3/__init__.py",
    "forge/action_teacher_clean_v3/__main__.py",
    "forge/action_teacher_clean_v3/contract.py",
    "forge/action_teacher_clean_v3/curriculum.py",
    "forge/action_teacher_clean_v3/recorder.py",
    "forge/action_teacher_v1/curriculum_v3.py",
    "forge/nature_sim_v2/demo.py",
)
ARRAY_NAMES=("frame","state","actor_state","actor_field","control","action","selected","timeline_event","timeline","counterfactual","tick")


def canonical(value):return (json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()


def source_sha256():
    digest=hashlib.sha256(b"nullvector-clean-cellular-action-teacher-v3\0")
    for relative in SOURCE_FILES:digest.update(relative.encode()+b"\0"+(PROJECT_ROOT/relative).read_bytes()+b"\0")
    return digest.hexdigest()
