from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from ..action_teacher_v1.contract import ACTIONS
from ..action_teacher_v1.curriculum_v3 import _bootstrap_four_slot_actor, _resolve
from ..action_teacher_v2 import extract_actor_features, extract_actor_field
from ..nature_counterfactual_nn import ACTIONS as COUNTERFACTUAL_ACTIONS
from ..nature_sim_v2.demo import NatureDemo, STUDENT_VIEW_HIDDEN
from ..nature_timeline_nn import EVENTS as TIMELINE_EVENTS, extract_world_features
from .contract import DEFAULT_ROOT
from .recorder import NaturalPlayRecorder, validate_trajectory


def _nearest_target(demo, actor):
    living=[item for item in demo.world.organisms.values() if item.alive and item.entity_id != actor.entity_id]
    return min(living,key=lambda item:(float(np.linalg.norm(demo.world._delta(actor.position,item.position))),item.entity_id)) if living else actor


def _record(demo, recorder, action, step):
    if not demo.student_view or any(getattr(demo,attribute) for attribute in STUDENT_VIEW_HIDDEN):raise RuntimeError("natural teacher overlay state drifted")
    actor=demo.world.organisms.get(demo.selected)
    if actor is None or not actor.alive:raise RuntimeError("natural teacher fixed actor died")
    demo.camera=actor.position.copy();control=demo._neural_control().copy();frame=demo.capture_clean_target();forecast=demo.timeline_forecast;timeline=np.asarray((forecast.confidence,forecast.population_delta,forecast.resource_delta),np.float32);counterfactual=np.asarray([[demo.counterfactuals[name].benefit,demo.counterfactuals[name].risk,demo.counterfactuals[name].population_delta,demo.counterfactuals[name].resource_delta] for name in COUNTERFACTUAL_ACTIONS],np.float32)
    recorder.append(frame=frame,state=extract_world_features(demo.world,demo.society),actor_state=extract_actor_features(demo.world,demo.selected),actor_field=extract_actor_field(demo.world,demo.selected),control=control,action=action,selected=demo.selected,timeline_event=TIMELINE_EVENTS.index(forecast.event),timeline=timeline,counterfactual=counterfactual,tick=demo.world.tick_index,episode_step=step)


def generate(*, root:Path=DEFAULT_ROOT, session_id:str, frames=1200, seed=0x4E41545552414C34, device="cuda"):
    if not 480 <= frames <= 3600:raise ValueError("natural teacher duration drifted")
    demo=NatureDemo(seed=seed,device=device,showcase=True);demo._set_student_view(True);actor_id=_bootstrap_four_slot_actor(demo);actor=demo.world.organisms[actor_id];actor.energy=1.15
    # Arrange the initial ecosystem once. Nothing is teleported during capture.
    others=[item for item in demo.world.organisms.values() if item.alive and item.entity_id != actor_id]
    for index,item in enumerate(others[:10]):
        angle=math.tau*index/max(1,min(10,len(others)));radius=3.5+(index%3)*2.2;item.position=(actor.position+np.asarray((math.cos(angle),math.sin(angle)))*radius)%demo.world.size;item.velocity*=0
    recorder=NaturalPlayRecorder(root,max_frames=frames+8);recorder.start(session_id,world_seed=demo.world.seed,tick=demo.world.tick_index);actions=tuple(name for name in ACTIONS if name!="none");counts={name:0 for name in ACTIONS};failures={name:0 for name in ACTIONS};action_period=12;warmup=24
    try:
        for step in range(frames):
            actor=demo.world.organisms.get(actor_id)
            if actor is None or not actor.alive:raise RuntimeError("natural teacher fixed actor died")
            phase=step/72.;demo.manual=np.asarray((math.cos(phase*.83),math.sin(phase*1.17)),np.float32)*(.36+.18*math.sin(phase*.31)**2);emitted="none"
            if step>=warmup and (step-warmup)%action_period==0:
                action=actions[((step-warmup)//action_period)%len(actions)];target=_nearest_target(demo,actor);demo.teacher_aim_override=target.position.copy()
                try:success=_resolve(demo,action,actor_id,target.entity_id,(step-warmup)//action_period)
                except (ValueError,RuntimeError,IndexError):success=False
                if success:emitted=action
                else:failures[action]+=1
            demo.action_latch=emitted;counts[emitted]+=1;demo.update(1/30);demo.selected=actor_id;_record(demo,recorder,emitted,step);demo.action_latch="none"
        destination=recorder.finish()
    finally:demo.neural_executor.shutdown(wait=True,cancel_futures=True);demo.pg.quit()
    manifest=validate_trajectory(destination);report={"format":"nullvector-natural-play-curriculum/4.0.0","session":session_id,"frames":manifest["frames"],"seed":seed,"fixed_actor":actor_id,"actions":counts,"failures":failures,"action_period":action_period,"protocol":"one fixed actor and camera; continuous controls; no staged teleports; all mechanisms live","trajectory_manifest_sha256":manifest["manifest_sha256"],"trajectory_arrays_sha256":manifest["arrays_sha256"]};(destination/"curriculum_report.json").write_text(json.dumps(report,sort_keys=True,indent=2)+"\n",encoding="utf-8");return report
