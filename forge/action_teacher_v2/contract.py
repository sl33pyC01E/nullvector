from __future__ import annotations

import hashlib
import json

from ..action_teacher_v1.contract import ACTIONS, COUNTERFACTUAL_SHAPE, FRAME_SIZE, STATE_FEATURES
from ..config import PROJECT_ROOT

FORMAT = "nullvector-cellular-action-teacher-trajectory/2.0.0"
ACTOR_FEATURES = 128
ACTOR_FIELD_SHAPE = (8, 32, 32)
SOURCE_FILES = (
    "forge/action_teacher_v2/contract.py",
    "forge/action_teacher_v2/actor.py",
    "forge/action_teacher_v2/recorder.py",
    "forge/action_teacher_v2/curriculum.py",
    "forge/action_teacher_v1/curriculum_v3.py",
)


def canonical(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def source_sha256() -> str:
    digest = hashlib.sha256(b"nullvector-cellular-action-teacher-v2\0")
    for relative in SOURCE_FILES:
        digest.update(relative.encode() + b"\0" + (PROJECT_ROOT / relative).read_bytes() + b"\0")
    return digest.hexdigest()
