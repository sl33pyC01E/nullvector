from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from ..action_teacher_v1.contract import ACTIONS
from ..action_teacher_v1.curriculum_v3 import _bootstrap_four_slot_actor, _resolve, _stage, spatial_schedule
from ..config import PROJECT_ROOT
from ..nature_counterfactual_nn import ACTIONS as COUNTERFACTUAL_ACTIONS
from ..nature_sim_v2.demo import NatureDemo
from ..nature_timeline_nn import EVENTS as TIMELINE_EVENTS, extract_world_features
from .actor import extract_actor_features, extract_actor_field
from .recorder import CellularActionTeacherRecorder, validate_trajectory


def _capture_teacher_frame(demo: NatureDemo) -> np.ndarray:
    """Render one raw world frame for the external v2 recorder.

    NatureDemo only populates ``teacher_frame`` when its internal v1 recorder or
    a neural presentation mode is active.  V2 deliberately owns a separate,
    richer recorder, so it must request the raw capture explicitly rather than
    toggling the v1 recorder and accidentally publishing two trajectories.
    """
    demo.draw()
    # The external recorder is not represented by NatureDemo.trajectory.active,
    # so draw() is allowed to leave a previous capture cached.  Recapture every
    # emitted step; otherwise the physiology changes while the pixel target is
    # silently frozen at frame zero.
    demo.teacher_frame = demo._capture_world_frame()
    frame = np.asarray(demo.teacher_frame)
    if frame.shape != (256, 256, 3) or frame.dtype != np.uint8:
        raise RuntimeError(f"cellular teacher frame contract violated: {frame.shape} {frame.dtype}")
    return np.ascontiguousarray(frame).copy()


def _emit(demo, recorder, action, manual, counts):
    demo.manual = manual.astype(np.float32); demo.action_latch = action; counts[action] += 1; demo.update(1 / 30)
    actor = demo.world.organisms.get(demo.selected)
    if actor is not None: demo.camera = actor.position.copy()
    control = demo._neural_control().copy(); frame = _capture_teacher_frame(demo); forecast = demo.timeline_forecast
    timeline = np.asarray((forecast.confidence, forecast.population_delta, forecast.resource_delta), np.float32)
    counterfactual = np.asarray([[demo.counterfactuals[name].benefit, demo.counterfactuals[name].risk, demo.counterfactuals[name].population_delta, demo.counterfactuals[name].resource_delta] for name in COUNTERFACTUAL_ACTIONS], np.float32)
    recorder.append(frame=frame, state=extract_world_features(demo.world, demo.society), actor_state=extract_actor_features(demo.world, demo.selected), actor_field=extract_actor_field(demo.world, demo.selected), control=control, action=action, selected=demo.selected, timeline_event=TIMELINE_EVENTS.index(forecast.event), timeline=timeline, counterfactual=counterfactual, tick=demo.world.tick_index)
    demo.action_latch = "none"


def generate(*, root: Path, session_id: str, repeats=6, seed=0x43454C4C554C4152, device="cuda"):
    schedule = spatial_schedule(repeats); demo = NatureDemo(seed=seed, device=device, showcase=True); four_slot_actor = _bootstrap_four_slot_actor(demo)
    recorder = CellularActionTeacherRecorder(root, max_frames=len(schedule) + 8); recorder.start(session_id, world_seed=demo.world.seed, tick=demo.world.tick_index)
    counts = {name: 0 for name in ACTIONS}; failures = {name: 0 for name in ACTIONS}; event_index = 0
    for repeat in range(repeats):
        for action in ACTIONS:
            actor_id, target_id = _stage(demo, action, event_index); phase = event_index * .61803398875; manual = np.asarray((math.cos(phase), math.sin(phase * .73))) * (.18 + repeat * .16)
            _emit(demo, recorder, "none", manual * .2, counts)
            try: success = _resolve(demo, action, actor_id, target_id, event_index)
            except (ValueError, RuntimeError, IndexError): success = False
            emitted = action if success else "none"
            if not success: failures[action] += 1
            _emit(demo, recorder, emitted, manual, counts); _emit(demo, recorder, "none", manual * .1, counts); event_index += 1
    destination = recorder.finish(); demo.neural_executor.shutdown(wait=True, cancel_futures=True); demo.pg.quit(); manifest = validate_trajectory(destination)
    missing = [name for name in ACTIONS if name != "none" and counts[name] < repeats]
    if missing: raise RuntimeError("cellular teacher missed actions: " + ",".join(missing))
    report = {"format": "nullvector-cellular-action-curriculum/2.0.0", "session": session_id, "frames": manifest["frames"], "repeats": repeats, "seed": seed, "actions": counts, "failures": failures, "protocol": "setup-none -> cellular spatial action -> settle-none; actor state and anatomy recorded", "four_slot_actor": four_slot_actor, "trajectory_manifest_sha256": manifest["manifest_sha256"], "trajectory_arrays_sha256": manifest["arrays_sha256"]}
    (destination / "curriculum_report.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"); return report


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "outputs/action_teacher_v2"); parser.add_argument("--session", required=True); parser.add_argument("--repeats", type=int, default=6); parser.add_argument("--seed", type=lambda value: int(value, 0), default=0x43454C4C554C4152); parser.add_argument("--device", default="cuda"); args = parser.parse_args(); print(json.dumps(generate(root=args.root, session_id=args.session, repeats=args.repeats, seed=args.seed, device=args.device), indent=2))


if __name__ == "__main__": main()
