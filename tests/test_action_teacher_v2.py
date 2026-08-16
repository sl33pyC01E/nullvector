from __future__ import annotations

import numpy as np

from forge.action_teacher_v2 import ACTOR_FEATURE_NAMES, ACTOR_FIELD_NAMES, CellularActionTeacherRecorder, extract_actor_features, extract_actor_field, validate_trajectory
from forge.action_teacher_v2.contract import ACTOR_FEATURES, ACTOR_FIELD_SHAPE
from forge.action_teacher_v2.curriculum import _capture_teacher_frame
from forge.nature_sim_v2.world import NatureWorld


def _world():
    world = NatureWorld(seed=0x43454C4C); world.seed_founders(variants_per_family=1); return world


def test_actor_features_expose_genetics_physiology_and_damage_state():
    world = _world(); entity = next(iter(world.organisms.values())); before = extract_actor_features(world, entity.entity_id); entity.body.impact((0, 0), 4, .6); after = extract_actor_features(world, entity.entity_id)
    assert len(ACTOR_FEATURE_NAMES) == ACTOR_FEATURES == 128
    assert before.shape == after.shape == (128,) and before.dtype == np.float32
    assert np.isfinite(before).all() and not np.array_equal(before, after)
    assert after[ACTOR_FEATURE_NAMES.index("system_integrity")] < before[ACTOR_FEATURE_NAMES.index("system_integrity")]


def test_actor_field_is_cellular_and_damage_local():
    world = _world(); entity = next(iter(world.organisms.values())); before = extract_actor_field(world, entity.entity_id); entity.body.cut((-8, 0), (8, 0), width=.7); after = extract_actor_field(world, entity.entity_id)
    assert tuple(ACTOR_FIELD_NAMES) == ("occupancy", "health", "fluid", "scar", "connected", "neural", "vital", "locomotor")
    assert before.shape == after.shape == ACTOR_FIELD_SHAPE and before.dtype == np.float16
    assert float(before[0].sum()) > 20 and float(after[1].sum()) < float(before[1].sum())


def test_missing_actor_has_explicit_zero_context():
    world = _world()
    assert not extract_actor_features(world, -1).any()
    assert not extract_actor_field(world, -1).any()


def test_cellular_teacher_roundtrips_actor_context(tmp_path):
    world = _world(); entity = next(iter(world.organisms.values())); recorder = CellularActionTeacherRecorder(tmp_path, max_frames=2); recorder.start("roundtrip", world_seed=world.seed, tick=0)
    recorder.append(frame=np.zeros((256, 256, 3), np.uint8), state=np.zeros(64, np.float32), actor_state=extract_actor_features(world, entity.entity_id), actor_field=extract_actor_field(world, entity.entity_id), control=np.zeros(4, np.float32), action="cut", selected=entity.entity_id, timeline_event=0, timeline=np.zeros(3, np.float32), counterfactual=np.zeros((5, 4), np.float32), tick=1)
    destination = recorder.finish(); manifest = validate_trajectory(destination)
    assert manifest["frames"] == 1
    assert manifest["shapes"]["actor_state"] == [1, 128]
    assert manifest["shapes"]["actor_field"] == [1, 8, 32, 32]
    with np.load(destination / manifest["artifact"]["path"], allow_pickle=False) as archive:
        assert archive["actor_field"].dtype == np.float16
        assert float(archive["actor_field"][0, 0].sum()) > 20


def test_external_recorder_explicitly_captures_raw_frame():
    class Demo:
        teacher_frame = np.zeros((256, 256, 3), np.uint8)

        def draw(self):
            pass

        def _capture_world_frame(self):
            return np.full((256, 256, 3), 17, np.uint8)

    frame = _capture_teacher_frame(Demo())
    assert frame.shape == (256, 256, 3)
    assert frame.dtype == np.uint8
    assert int(frame[0, 0, 0]) == 17
