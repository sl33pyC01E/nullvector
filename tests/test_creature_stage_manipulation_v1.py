from __future__ import annotations

import numpy as np

from forge.creature_stage_developmental.development import develop
from forge.creature_stage_developmental.genomes import review_genomes
from forge.creature_stage_manipulation_v1 import NeuralManipulationArena
from forge.creature_stage_manipulation_v1.articulation import ArticulatedBody
from forge.creature_stage_neural_grasper_v1.feeding import FoodClump, feeder_status


def _food(position) -> FoodClump:
    return FoodClump(np.asarray(position, dtype=np.float64), np.zeros(2), 1.0, .45, 1.0, (1, 1, 1, 1, 1))


def test_humanoid_graspers_are_locomotor_peer_limbs() -> None:
    organism = develop(review_genomes()[0])
    articulation = ArticulatedBody.from_organism(organism)
    articulation.require_peer_limbs({"arm"}, {"leg"})
    arms = [articulation.geometry(index) for index, gene in enumerate(organism.genome.appendages) if gene.kind == "arm"]
    legs = [articulation.geometry(index) for index, gene in enumerate(organism.genome.appendages) if gene.kind == "leg"]
    assert {limb.segments for limb in arms} == {limb.segments for limb in legs} == {2}
    assert .90 < np.mean([limb.length for limb in arms]) / np.mean([limb.length for limb in legs]) < 1.0
    assert .80 < np.mean([limb.cell_count for limb in arms]) / np.mean([limb.cell_count for limb in legs]) < 1.20


def test_grasper_chain_has_inertia_and_preserves_bone_lengths() -> None:
    articulation = ArticulatedBody.from_organism(develop(review_genomes()[0]))
    appendage = 0
    before = articulation.endpoint(appendage).copy()
    target = articulation.root(appendage) + np.asarray((-7.0, -1.5))
    first = articulation.solve(appendage, target, .65)
    assert np.linalg.norm(first - before) > .05
    assert np.linalg.norm(first - target) > .05  # no kinematic pose snap
    for _ in range(80):
        final = articulation.solve(appendage, target, .65)
    assert np.linalg.norm(final - target) < .35
    assert np.linalg.norm(articulation.velocities[articulation.chain_ids[appendage][1:]], axis=1).max() > 0
    assert articulation.max_length_error() < 1e-4


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
        assert arena.articulation.max_length_error() < 1e-4, genome.genome_id


def test_neural_throw_releases_only_an_attached_target_with_recoil() -> None:
    organism = develop(review_genomes()[8])
    arena = NeuralManipulationArena(organism, device="cpu")
    target_id = arena.add_clump(_food((12.0, 0.0)), cohesion=2.0, impact_mode="bounce")
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
    release_x = float(arena.targets[target_id].position[0])
    peak_height = arena.target_kinetics[target_id].height
    for _ in range(150):
        arena.integrate_free_target(target_id, .05)
        peak_height = max(peak_height, arena.target_kinetics[target_id].height)
    assert float(arena.targets[target_id].position[0]) - release_x > 20.0
    assert peak_height > 8.0 and arena.target_kinetics[target_id].impacts > 1


def test_2p5d_impacts_bounce_roll_or_thud() -> None:
    distances = {}
    angles = {}
    for mode in ("bounce", "roll", "thud"):
        arena = NeuralManipulationArena(develop(review_genomes()[0]), device="cpu")
        target_id = arena.add_clump(_food((0.0, 0.0)), impact_mode=mode)
        target = arena.targets[target_id]
        target.velocity[:] = (7.0, 0.0)
        kinetics = arena.target_kinetics[target_id]
        kinetics.height = 5.5
        kinetics.vertical_velocity = 9.5
        rebound_height = 0.0
        for _ in range(200):
            arena.integrate_free_target(target_id, .05)
            if kinetics.impacts:
                rebound_height = max(rebound_height, kinetics.height)
        distances[mode] = float(target.position[0])
        angles[mode] = abs(float(kinetics.angle))
        assert kinetics.impacts >= 1 and kinetics.height >= 0
        if mode == "bounce":
            assert rebound_height > 2.0 and kinetics.impacts > 1
        else:
            assert kinetics.impacts == 1 and kinetics.vertical_velocity == 0
    assert distances["bounce"] > distances["roll"] > distances["thud"] + 10.0
    assert angles["roll"] > .5
    assert abs(arena.target_kinetics[target_id].angular_velocity) < 1e-4  # final thud case


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
