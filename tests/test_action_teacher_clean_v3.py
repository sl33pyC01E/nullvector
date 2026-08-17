from __future__ import annotations

import numpy as np

from forge.action_teacher_clean_v3 import CleanCellularTeacherRecorder,validate_trajectory
from forge.action_teacher_clean_v3.contract import ACTOR_FIELD_SHAPE,COUNTERFACTUAL_SHAPE,FRAME_SIZE


def test_clean_teacher_roundtrip_and_view_contract(tmp_path):
    recorder=CleanCellularTeacherRecorder(tmp_path,max_frames=32);recorder.start("clean-test",world_seed=7,tick=0)
    for tick in range(1,33):
        recorder.append(frame=np.zeros((FRAME_SIZE[1],FRAME_SIZE[0],3),np.uint8),state=np.zeros(64,np.float32),actor_state=np.zeros(128,np.float32),actor_field=np.zeros(ACTOR_FIELD_SHAPE,np.float16),control=np.zeros(4,np.float32),action="none",selected=1,timeline_event=0,timeline=np.zeros(3,np.float32),counterfactual=np.zeros(COUNTERFACTUAL_SHAPE,np.float32),tick=tick)
    manifest=validate_trajectory(recorder.finish());assert manifest["frames"]==32 and manifest["view_contract"]["clean_student_view"] and not manifest["view_contract"]["diagnostic_overlays"]


def test_clean_teacher_rejects_duplicate_ticks(tmp_path):
    recorder=CleanCellularTeacherRecorder(tmp_path,max_frames=32);recorder.start("duplicate",world_seed=9,tick=0);kwargs={"frame":np.zeros((256,256,3),np.uint8),"state":np.zeros(64,np.float32),"actor_state":np.zeros(128,np.float32),"actor_field":np.zeros(ACTOR_FIELD_SHAPE,np.float16),"control":np.zeros(4,np.float32),"action":"none","selected":1,"timeline_event":0,"timeline":np.zeros(3,np.float32),"counterfactual":np.zeros(COUNTERFACTUAL_SHAPE,np.float32),"tick":1};recorder.append(**kwargs)
    try:recorder.append(**kwargs)
    except ValueError as exc:assert "ticks" in str(exc)
    else:raise AssertionError("duplicate tick accepted")
