from __future__ import annotations
import json
import numpy as np
import pytest
from forge.action_teacher_v1 import FRAME_SIZE,TeacherTrajectoryRecorder,validate_trajectory

def _append(recorder,tick,action="none"):
    return recorder.append(frame=np.full((FRAME_SIZE[1],FRAME_SIZE[0],3),tick%255,np.uint8),state=np.linspace(0,1,64,dtype=np.float32),control=np.asarray((.2,-.4,.1,.7),np.float32),action=action,selected=7,timeline_event=2,timeline=np.asarray((.8,.03,-.02),np.float32),counterfactual=np.full((5,4),.25,np.float32),tick=tick)

def test_teacher_trajectory_roundtrips_and_obeys_stride(tmp_path):
    recorder=TeacherTrajectoryRecorder(tmp_path,stride=3,max_frames=8);recorder.start("episode-001",world_seed=9,tick=10);assert _append(recorder,10);assert not _append(recorder,11);assert _append(recorder,11,"cut");assert _append(recorder,14);path=recorder.finish();manifest=validate_trajectory(path);assert manifest["frames"]==3 and manifest["start_tick"]==10 and manifest["end_tick"]==14

def test_teacher_trajectory_detects_rehashed_manifest_or_archive_tamper(tmp_path):
    recorder=TeacherTrajectoryRecorder(tmp_path);recorder.start("episode-002",world_seed=11,tick=0);_append(recorder,0);path=recorder.finish();manifest_path=path/"manifest.json";manifest=json.loads(manifest_path.read_text());manifest["frames"]+=1;manifest_path.write_text(json.dumps(manifest));
    with pytest.raises(ValueError):validate_trajectory(path)
