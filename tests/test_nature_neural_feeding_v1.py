from __future__ import annotations

import numpy as np
from pathlib import Path

from forge.creature_stage_developmental import develop
from forge.nature_neural_feeding_v1 import NatureNeuralFeedingSystem
from forge.nature_neural_feeding_v1.system import _controller_view
from forge.nature_sim_v2 import NatureWorld, founder_genomes, load_world, save_world
from forge.nature_sim_v2.grafting import graft_appendage_pair


MATERIALS = ("biomass", "flora", "flora", "phase", "mineral")


def _single(family: int, seed: int = 71):
    system = NatureNeuralFeedingSystem(seed=seed, device="cpu")
    world = NatureWorld(seed=seed + 1, size=32, max_population=20, feeding_system=system)
    genome = founder_genomes(variants_per_family=1)[family]
    entity_id = world.add_organism(genome, (16.0, 16.0), energy=.20)
    entity = world.organisms[entity_id]
    entity.reserve = 0
    system.add_clump((17.0, 16.0), material=MATERIALS[family], mass=1.0)
    return world, system, entity


def test_all_families_use_neural_grasper_and_physical_feeder() -> None:
    for family in range(5):
        world, system, entity = _single(family, 80 + family)
        for _ in range(450):
            world.step(.05, publish=False)
            if system.absorbed_mass > .08:
                break
        assert system.grasps > 0, family
        assert system.absorbed_mass > .08, family
        assert entity.reserve > 0, family


def test_evolved_anatomy_is_projected_without_removing_living_appendages() -> None:
    founders = founder_genomes()
    anomaly = next(item for item in founders if item.family == 3 and len(item.developmental.appendages) == 8)
    humanoid = next(item for item in founders if item.family == 0)
    arm = next(item.appendage_id for item in humanoid.developmental.appendages if item.kind == "arm")
    grafted = graft_appendage_pair(anomaly, humanoid, arm, seed=0xC011EC7)
    organism = develop(grafted.developmental)
    projected, lookup = _controller_view(organism)
    assert len(organism.genome.appendages) == 10
    assert len(projected.genome.appendages) == 8
    assert len(lookup) == 8 and len(set(lookup)) == 8
    assert any(gene.kind == "arm" for gene in projected.genome.appendages)
    assert organism.genome.appendages == grafted.developmental.appendages


def test_persistent_held_clump_is_exactly_constrained_to_the_hand() -> None:
    world, system, entity = _single(0, 88)
    attached_frames = 0
    for _ in range(500):
        world.step(.05, publish=False)
        state = system.entities[entity.entity_id]
        if not state.constraint.attached or state.target_id not in system.clumps:
            continue
        clump = system.clumps[int(state.target_id)]
        appendage = int(state.grasp_appendage)
        expected = state.articulation.endpoint(appendage) / 12.0
        actual = world._delta(entity.position, clump.food.position)
        assert np.linalg.norm(actual - expected) < 1e-6
        attached_frames += 1
        if attached_frames >= 24:
            break
    assert attached_frames >= 24


def test_consumed_clump_is_retired_before_grasp_physics() -> None:
    world, system, entity = _single(0, 93)
    clump_id = max(system.clumps)
    system.clumps[clump_id].food.mass = 1e-6
    state = system._entity(entity)
    state.target_id = clump_id
    state.constraint.attached = True
    result = system.step_entity(world, entity, .05)
    assert result == {"contact": False, "absorbed": 0.0, "attached": False, "target": -1}
    assert clump_id not in system.clumps


def test_predation_produces_tangible_matter_not_instant_predator_energy() -> None:
    world, system, predator = _single(0)
    prey_genome = founder_genomes(variants_per_family=1)[2]
    prey_id = world.add_organism(prey_genome, tuple(predator.position + (.2, 0)), energy=.6)
    prey = world.organisms[prey_id]
    before = predator.energy
    system.on_predation(world, predator, prey, .12)
    assert predator.energy == before
    produced = [item for item in system.clumps.values() if item.source == f"injury:{prey_id}"]
    assert len(produced) == 1 and produced[0].food.material == "flora" and produced[0].food.mass > 0


def test_persistent_world_clumps_have_2p5d_material_impacts() -> None:
    world, system, _ = _single(0, 97)
    modes = (("phase", "bounce"), ("mineral", "roll"), ("biomass", "thud"))
    clumps = []
    for offset, (material, mode) in enumerate(modes):
        clump_id = system.add_clump((2.0, 4.0 + offset), material=material, impact_mode=mode)
        system.throw_clump(clump_id, (4.0, 0.0), height=.46, vertical_velocity=3.1)
        clumps.append(system.clumps[clump_id])
    rebound = 0.0
    for _ in range(220):
        system.step_environment(world, .05)
        if clumps[0].impacts:
            rebound = max(rebound, clumps[0].height)
    assert clumps[0].impacts > 1 and rebound > .3
    assert clumps[1].impacts == 1 and abs(clumps[1].angle) > .5
    assert clumps[2].impacts == 1 and np.linalg.norm(clumps[2].food.velocity) < 1e-5
    assert system.throws == 3


def test_thrown_inert_mineral_damages_once_and_positive_matter_aids() -> None:
    world, system, entity = _single(0, 109)
    mineral_id = system.add_clump((14.6, 16.0), material="mineral", mass=1.2, source="throw-test")
    system.throw_clump(mineral_id, (8.0, 0.0), height=0.0, vertical_velocity=0.0)
    health_before = float(entity.body.health.mean())
    system._integrate_clump(system.clumps[mineral_id], world, .1)
    health_after_impact = float(entity.body.health.mean())
    assert health_after_impact < health_before
    impacts = [event for event in world.events if event["type"] == "material_impact"]
    assert len(impacts) == 1 and impacts[0]["material"] == "mineral"
    for _ in range(8):
        system._integrate_clump(system.clumps[mineral_id], world, .05)
    assert len([event for event in world.events if event["type"] == "material_impact"]) == 1

    entity.body.impact((0.0, 0.0), 5.0, .25)
    wounded = float(entity.body.health.mean())
    biomass_id = system.add_clump((14.6, 16.0), material="biomass", mass=1.2, source="aid-test")
    system.throw_clump(biomass_id, (8.0, 0.0), height=0.0, vertical_velocity=0.0)
    energy_before = entity.energy
    system._integrate_clump(system.clumps[biomass_id], world, .1)
    assert float(entity.body.health.mean()) > wounded
    assert entity.energy > energy_before
    aid = [event for event in world.events if event["type"] == "material_aid"]
    assert len(aid) == 1 and aid[0]["material"] == "biomass"


def test_neural_feeding_world_replay_is_exact() -> None:
    hashes = []
    for _ in range(2):
        world, system, _ = _single(1, 101)
        for _ in range(120):
            world.step(.05, publish=False)
        hashes.append(world.snapshot().semantic_sha256)
    assert hashes[0] == hashes[1]


def test_legacy_world_keeps_existing_abstract_path() -> None:
    world = NatureWorld(seed=13, size=32, max_population=20)
    entity_id = world.add_organism(founder_genomes(variants_per_family=1)[0], (8, 8), energy=.2)
    before = world.organisms[entity_id].energy
    world._consume(world.organisms[entity_id], .25)
    assert world.organisms[entity_id].energy >= before


def test_neural_feeding_save_load_preserves_exact_world(tmp_path: Path) -> None:
    world, system, _ = _single(0, 141)
    ballistic_id = system.add_clump((9.0, 9.0), material="mineral")
    system.throw_clump(ballistic_id, (2.5, -.4), height=.8, vertical_velocity=2.7)
    for _ in range(80):
        world.step(.05, publish=False)
    before = world.snapshot().semantic_sha256
    path = tmp_path / "feeding-world.npz"
    save_world(world, path)
    restored_system = NatureNeuralFeedingSystem(seed=141, device="cpu")
    restored = load_world(path, feeding_system=restored_system)
    assert restored.snapshot().semantic_sha256 == before
    assert restored_system.semantic_sha256() == system.semantic_sha256()
