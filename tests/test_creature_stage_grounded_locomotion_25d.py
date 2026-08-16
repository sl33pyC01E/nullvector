from __future__ import annotations

import numpy as np

from forge.creature_stage_developmental import develop
from forge.creature_stage_grounded_locomotion_25d import Grounded25DConfig, simulate_25d
from forge.creature_stage_morphology_v2 import morphology_review_genomes


def _short_config()->Grounded25DConfig:
    return Grounded25DConfig(frames=60,substeps=2,constraint_iterations=5)


def test_depth_and_lateral_control_use_persistent_ground_contacts() -> None:
    animal=develop(morphology_review_genomes()[6])
    rollout=simulate_25d(animal,_short_config())
    assert rollout.primary_mode=="step"
    assert rollout.lateral_extent>.1 and rollout.depth_extent>.1
    assert rollout.contact_switches>=4
    assert rollout.appendage_motion_px>.1
    assert rollout.maximum_chassis_tilt_degrees<8
    assert rollout.maximum_edge_strain<.4


def test_sprite_chassis_does_not_rotate_with_ground_heading() -> None:
    humanoid=develop(morphology_review_genomes()[0])
    rollout=simulate_25d(humanoid,_short_config())
    headings=np.stack([frame.heading for frame in rollout.frames])
    assert np.ptp(headings[:,0])>1.5 and np.ptp(headings[:,1])>1.5
    assert rollout.maximum_chassis_tilt_degrees<8


def test_family_specific_locomotor_substrates() -> None:
    genomes=morphology_review_genomes()
    modes=[simulate_25d(develop(genomes[index]),_short_config()).primary_mode for index in (0,6,12,18,24)]
    assert modes==["step","step","drag","float","wheel"]


def test_rollout_is_exactly_replayable() -> None:
    organism=develop(morphology_review_genomes()[24])
    left=simulate_25d(organism,_short_config());right=simulate_25d(organism,_short_config())
    assert left.identity_sha256==right.identity_sha256
    for a,b in zip(left.frames,right.frames):
        assert np.array_equal(a.ground_position,b.ground_position)
        assert np.array_equal(a.nodes_local,b.nodes_local)
