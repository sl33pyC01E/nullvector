from __future__ import annotations

import numpy as np
import torch

from forge.creature_stage_neural_grasper_v1.constraint import GraspBody, GraspConstraint, solve_grasp
from forge.creature_stage_neural_grasper_v1.feeding import FoodClump, FeedingState, absorb_food, feeder_status, metabolize_reserve
from forge.creature_stage_neural_grasper_v1.contract import GLOBAL_FEATURES, MAX_APPENDAGES, OWNER_FEATURES, TARGET_FEATURES, ModelConfig
from forge.creature_stage_neural_grasper_v1.dataset import build_corpus
from forge.creature_stage_neural_grasper_v1.model import NeuralGrasperController
from forge.creature_stage_developmental.development import develop
from forge.creature_stage_developmental.genomes import review_genomes
from forge.living_body_substrate import LivingBody


def test_corpus_is_deterministic_and_family_complete() -> None:
    first = build_corpus(split="validation", cases_per_identity=32); second = build_corpus(split="validation", cases_per_identity=32)
    assert first.semantic_sha256 == second.semantic_sha256
    assert set(first.family.tolist()) == set(range(5))
    assert first.owner_meta.shape[1:] == (MAX_APPENDAGES, OWNER_FEATURES)
    assert first.target.shape[1:] == (TARGET_FEATURES,)


def test_model_outputs_physical_command_shapes() -> None:
    model = NeuralGrasperController(ModelConfig(width=128, depth=3))
    owner = torch.randn(3, MAX_APPENDAGES, OWNER_FEATURES); mask = torch.ones(3, MAX_APPENDAGES, dtype=torch.bool)
    result = model(owner, mask, torch.randn(3, TARGET_FEATURES), torch.randn(3, GLOBAL_FEATURES))
    assert result.appendage_logits.shape == (3, MAX_APPENDAGES)
    assert result.reach.shape == (3, 2)
    assert result.throw_impulse.shape == (3, 2)
    assert bool(torch.isfinite(result.force).all()) and bool(((result.force >= 0) & (result.force <= 1)).all())


def test_constraint_conserves_pair_impulse_and_releases() -> None:
    body = GraspBody(np.asarray((0.0, 0.0)), np.zeros(2), 2.0); target = GraspBody(np.asarray((.8, 0.0)), np.zeros(2), 1.0); state = GraspConstraint()
    momentum_before = body.velocity * body.mass + target.velocity * target.mass
    result = solve_grasp(body, target, effector=np.asarray((.2, 0.0)), engage=True, force=.5, brace=0, cohesion=1, state=state, delta=.1)
    momentum_after = body.velocity * body.mass + target.velocity * target.mass
    assert result["attached"] and np.allclose(momentum_before, momentum_after)
    released = solve_grasp(body, target, effector=np.asarray((.2, 0.0)), engage=False, force=0, brace=0, cohesion=1, state=state, delta=.1)
    assert not released["attached"]


def test_weak_material_tears_under_strain() -> None:
    body = GraspBody(np.asarray((0.0, 0.0)), np.zeros(2), 2.0); target = GraspBody(np.asarray((1.2, 0.0)), np.zeros(2), .5); state = GraspConstraint(attached=True)
    result = solve_grasp(body, target, effector=np.asarray((0.0, 0.0)), engage=True, force=1, brace=.8, cohesion=.1, state=state, delta=.1)
    assert result["torn"] and not result["attached"]


def test_throw_conserves_momentum_and_requires_attachment() -> None:
    body = GraspBody(np.zeros(2), np.zeros(2), 2.0); target = GraspBody(np.ones(2), np.zeros(2), .5)
    state = GraspConstraint(attached=True)
    momentum_before = body.velocity * body.mass + target.velocity * target.mass
    result = solve_grasp(body, target, effector=np.zeros(2), engage=False, force=0, brace=0, cohesion=1, state=state, delta=.1, release_impulse=np.asarray((1.4, -.3)))
    momentum_after = body.velocity * body.mass + target.velocity * target.mass
    assert result["thrown"] and np.allclose(momentum_before, momentum_after)
    idle_velocity = target.velocity.copy()
    result = solve_grasp(body, target, effector=np.zeros(2), engage=False, force=0, brace=0, cohesion=1, state=state, delta=.1, release_impulse=np.asarray((1.4, -.3)))
    assert not result["thrown"] and np.array_equal(target.velocity, idle_velocity)


def test_every_family_has_live_feeder_and_digestive_route() -> None:
    observed = set()
    for genome in review_genomes():
        body = LivingBody(develop(genome))
        status = feeder_status(body)
        observed.add(body.family)
        assert status.kind in ("mouth", "root_feeder", "transmuter_aperture", "fuel_port")
        assert status.live_feeder_cells > 0 and status.live_digestive_cells > 0
        assert status.route_intact and status.capacity > .9
    assert observed == set(range(5))


def test_food_requires_physical_feeder_contact_and_intact_route() -> None:
    body = LivingBody(develop(review_genomes()[2]))  # base animalian
    status = feeder_status(body); mouth = body.organism.cell_xy[status.feeder_mask][0].astype(np.float64)
    feeding = FeedingState(reserve=0, fullness_seconds=0)
    far = FoodClump(mouth + 20, np.zeros(2), 1, .4, 1, (1, 1, 1, 1, 1))
    result = absorb_food(body, feeding, far, body_position=np.zeros(2), delta=.25)
    assert not result.contacted and result.absorbed_mass == 0 and feeding.reserve == 0
    food = FoodClump(mouth.copy(), np.zeros(2), 1, .4, 1, (1, 1, 1, 1, 1))
    result = absorb_food(body, feeding, food, body_position=np.zeros(2), delta=.25)
    assert result.contacted and result.route_intact and result.absorbed_mass > 0 and feeding.reserve > 0
    body.health[~(status.feeder_mask | status.digestive_mask)] = 0
    isolated = feeder_status(body)
    assert not isolated.route_intact
    before = food.mass
    result = absorb_food(body, feeding, food, body_position=np.zeros(2), delta=.25)
    assert result.contacted and not result.route_intact and result.absorbed_mass == 0 and food.mass == before


def test_fullness_reserve_is_long_lived_and_energy_buffered() -> None:
    body = LivingBody(develop(review_genomes()[0]))
    feeding = FeedingState(reserve=2, fullness_seconds=120)
    body.energy = .4
    released = metabolize_reserve(body, feeding, delta=1, activity=.5)
    assert 0 < released < .01
    assert feeding.fullness_seconds == 119 and feeding.reserve > 1.99 and body.energy > .4
