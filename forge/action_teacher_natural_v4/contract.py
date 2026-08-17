from __future__ import annotations

import hashlib
import json

from ..action_teacher_v1.contract import ACTIONS, COUNTERFACTUAL_SHAPE, FRAME_SIZE, STATE_FEATURES
from ..action_teacher_v2.contract import ACTOR_FEATURES, ACTOR_FIELD_SHAPE
from ..config import PROJECT_ROOT


FORMAT = "nullvector-natural-play-cellular-teacher/4.0.0"
DEFAULT_ROOT = PROJECT_ROOT / "outputs/action_teacher_natural_v4/production_v1"
SOURCE_FILES = (
    "forge/action_teacher_natural_v4/__init__.py",
    "forge/action_teacher_natural_v4/__main__.py",
    "forge/action_teacher_natural_v4/contract.py",
    "forge/action_teacher_natural_v4/curriculum.py",
    "forge/action_teacher_natural_v4/recorder.py",
    "forge/action_teacher_v1/curriculum_v3.py",
    "forge/nature_sim_v2/demo.py",
    "forge/nature_neural_feeding_v1/system.py",
)
ARRAY_NAMES = ("frame", "state", "actor_state", "actor_field", "control", "action", "selected", "timeline_event", "timeline", "counterfactual", "tick", "episode_step")


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def source_sha256():
    digest = hashlib.sha256(b"nullvector-natural-play-cellular-teacher-v4\0")
    for relative in SOURCE_FILES:
        digest.update(relative.encode() + b"\0" + (PROJECT_ROOT / relative).read_bytes() + b"\0")
    return digest.hexdigest()
