from __future__ import annotations

import inspect

import numpy as np

from forge.action_teacher_natural_v4 import NaturalPlayRecorder, validate_trajectory
from forge.action_teacher_natural_v4.contract import ACTOR_FEATURES, ACTOR_FIELD_SHAPE, COUNTERFACTUAL_SHAPE, FRAME_SIZE, STATE_FEATURES
from forge.action_teacher_natural_v4.curriculum import generate


def test_natural_teacher_roundtrip_has_one_actor_and_no_camera_cuts(tmp_path) -> None:
    recorder=NaturalPlayRecorder(tmp_path,max_frames=128);recorder.start("natural-test",world_seed=7,tick=10)
    for step in range(128):
        recorder.append(frame=np.zeros((FRAME_SIZE[1],FRAME_SIZE[0],3),np.uint8),state=np.zeros(STATE_FEATURES,np.float32),actor_state=np.zeros(ACTOR_FEATURES,np.float32),actor_field=np.zeros(ACTOR_FIELD_SHAPE,np.float16),control=np.zeros(4,np.float32),action="none",selected=4,timeline_event=0,timeline=np.zeros(3,np.float32),counterfactual=np.zeros(COUNTERFACTUAL_SHAPE,np.float32),tick=11+step,episode_step=step)
    manifest=validate_trajectory(recorder.finish())
    assert manifest["frames"]==128
    assert manifest["view_contract"]["fixed_actor_camera"]
    assert manifest["view_contract"]["no_staged_camera_cuts"]


def test_natural_curriculum_never_calls_staged_teleport_helper() -> None:
    source=inspect.getsource(generate)
    assert "_stage(" not in source
    assert "demo.selected=actor_id" in source
    assert "_resolve(" in source
