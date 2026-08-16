from __future__ import annotations

import numpy as np

from forge.creature_stage_developmental.development import develop
from forge.creature_stage_developmental.genomes import review_genomes
from forge.creature_stage_manipulation_v1 import NeuralManipulationArena
from forge.creature_stage_neural_grasper_v1.feeding import FoodClump, feeder_status


def _food(position) -> FoodClump:
    return FoodClump(np.asarray(position, dtype=np.float64), np.zeros(2), 1.0, .45, 1.0, (1, 1, 1, 1, 1))


def test_neural_closed_loop_grasps_carries_and_physically_feeds() -> None:
    for genome in review_genomes()[::2]:
        arena = NeuralManipulationArena(develop(genome), device="cpu")
        arena.feeding.reserve = 0
        arena.feeding.fullness_seconds = 0
        target_id = arena.add_clump(_food((10.0, 1.0)))
        attached_seen = contact_seen = False
        for _ in range(800):
            step = arena.step(target_id, goal="consume", delta=.05)
            attached_seen |= step.attached
            contact_seen |= step.feeder_contact
            if arena.feeding.consumed_mass > .15:
                break
        assert attached_seen and contact_seen, genome.genome_id
        assert arena.feeding.consumed_mass > .15 and arena.feeding.reserve > .15, genome.genome_id
        assert arena.targets[target_id].mass < .85, genome.genome_id


def test_neural_throw_releases_only_an_attached_target_with_recoil() -> None:
    organism = develop(review_genomes()[8])
    arena = NeuralManipulationArena(organism, device="cpu")
    target_id = arena.add_clump(_food((12.0, 0.0)), cohesion=2.0)
    for _ in range(500):
        step = arena.step(target_id, goal="carry", delta=.05)
        if step.attached:
            break
    assert arena.constraint.attached
    momentum_before = arena.body.velocity * arena.body.mass + arena.targets[target_id].velocity * arena.targets[target_id].mass
    step = arena.step(target_id, goal="throw", delta=.05, throw_strength=1.0)
    assert step.thrown and not arena.constraint.attached
    assert arena.targets[target_id].velocity[0] > 0
    assert arena.body.velocity[0] < 0
    # Neural bracing is allowed to exchange the residual recoil with ground,
    # but cannot create more pair momentum than the commanded impulse.
    momentum_after = arena.body.velocity * arena.body.mass + arena.targets[target_id].velocity * arena.targets[target_id].mass
    assert float(np.linalg.norm(momentum_after - momentum_before)) < 6.1


def test_live_feeder_damage_still_blocks_arena_absorption() -> None:
    organism = develop(review_genomes()[2])
    arena = NeuralManipulationArena(organism, device="cpu")
    status = feeder_status(arena.living)
    arena.living.health[status.feeder_mask] = 0
    mouth = organism.cell_xy[status.feeder_mask][0]
    target_id = arena.add_clump(_food(mouth))
    before = arena.targets[target_id].mass
    for _ in range(20):
        arena.step(target_id, goal="consume", delta=.05)
    assert arena.targets[target_id].mass == before
    assert arena.feeding.consumed_mass == 0
