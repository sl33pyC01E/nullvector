from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from ..action_teacher_v1.contract import ACTIONS
from ..action_teacher_v1.curriculum_v3 import _bootstrap_four_slot_actor,_resolve,_stage,spatial_schedule
from ..action_teacher_v2 import extract_actor_features,extract_actor_field
from ..nature_counterfactual_nn import ACTIONS as COUNTERFACTUAL_ACTIONS
from ..nature_sim_v2.demo import NatureDemo,STUDENT_VIEW_HIDDEN
from ..nature_timeline_nn import EVENTS as TIMELINE_EVENTS,extract_world_features
from .contract import DEFAULT_ROOT
from .recorder import CleanCellularTeacherRecorder,validate_trajectory


def _emit(demo,recorder,action,manual,counts):
    if not demo.student_view or any(getattr(demo,attribute) for attribute in STUDENT_VIEW_HIDDEN):raise RuntimeError("clean teacher overlay state drifted")
    demo.manual=manual.astype(np.float32);demo.action_latch=action;counts[action]+=1;demo.update(1/30);actor=demo.world.organisms.get(demo.selected)
    if actor is not None:demo.camera=actor.position.copy()
    control=demo._neural_control().copy();frame=demo.capture_clean_target();forecast=demo.timeline_forecast;timeline=np.asarray((forecast.confidence,forecast.population_delta,forecast.resource_delta),np.float32);counterfactual=np.asarray([[demo.counterfactuals[name].benefit,demo.counterfactuals[name].risk,demo.counterfactuals[name].population_delta,demo.counterfactuals[name].resource_delta] for name in COUNTERFACTUAL_ACTIONS],np.float32)
    recorder.append(frame=frame,state=extract_world_features(demo.world,demo.society),actor_state=extract_actor_features(demo.world,demo.selected),actor_field=extract_actor_field(demo.world,demo.selected),control=control,action=action,selected=demo.selected,timeline_event=TIMELINE_EVENTS.index(forecast.event),timeline=timeline,counterfactual=counterfactual,tick=demo.world.tick_index);demo.action_latch="none"


def generate(*,root:Path=DEFAULT_ROOT,session_id:str,repeats=6,seed=0x434C45414E5633,device="cuda"):
    schedule=spatial_schedule(repeats);demo=NatureDemo(seed=seed,device=device,showcase=True);demo._set_student_view(True);four_slot_actor=_bootstrap_four_slot_actor(demo);recorder=CleanCellularTeacherRecorder(root,max_frames=len(schedule)+8);recorder.start(session_id,world_seed=demo.world.seed,tick=demo.world.tick_index);counts={name:0 for name in ACTIONS};failures={name:0 for name in ACTIONS};event_index=0
    try:
        for repeat in range(repeats):
            for action in ACTIONS:
                actor_id,target_id=_stage(demo,action,event_index);phase=event_index*.61803398875;manual=np.asarray((math.cos(phase),math.sin(phase*.73)))*(.18+repeat*.16);_emit(demo,recorder,"none",manual*.2,counts)
                try:success=_resolve(demo,action,actor_id,target_id,event_index)
                except (ValueError,RuntimeError,IndexError):success=False
                emitted=action if success else "none"
                if not success:failures[action]+=1
                _emit(demo,recorder,emitted,manual,counts);_emit(demo,recorder,"none",manual*.1,counts);event_index+=1
        destination=recorder.finish()
    finally:demo.neural_executor.shutdown(wait=True,cancel_futures=True);demo.pg.quit()
    manifest=validate_trajectory(destination);missing=[name for name in ACTIONS if name!="none" and counts[name]<repeats]
    if missing:raise RuntimeError("clean teacher missed actions: "+",".join(missing))
    report={"format":"nullvector-clean-cellular-action-curriculum/3.0.0","session":session_id,"frames":manifest["frames"],"repeats":repeats,"seed":seed,"actions":counts,"failures":failures,"protocol":"overlay-free student view; setup-none -> cellular action -> settle-none; contiguous actor+cell authority","four_slot_actor":four_slot_actor,"trajectory_manifest_sha256":manifest["manifest_sha256"],"trajectory_arrays_sha256":manifest["arrays_sha256"]};(destination/"curriculum_report.json").write_text(json.dumps(report,sort_keys=True,indent=2)+"\n",encoding="utf-8");return report
