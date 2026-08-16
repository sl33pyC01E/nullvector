from __future__ import annotations

import numpy as np
from pathlib import Path

from forge.nature_neural_feeding_v1 import NatureNeuralFeedingSystem
from forge.nature_sim_v2 import NatureWorld, founder_genomes, load_world, save_world


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
    for _ in range(80):
        world.step(.05, publish=False)
    before = world.snapshot().semantic_sha256
    path = tmp_path / "feeding-world.npz"
    save_world(world, path)
    restored_system = NatureNeuralFeedingSystem(seed=141, device="cpu")
    restored = load_world(path, feeding_system=restored_system)
    assert restored.snapshot().semantic_sha256 == before
    assert restored_system.semantic_sha256() == system.semantic_sha256()
