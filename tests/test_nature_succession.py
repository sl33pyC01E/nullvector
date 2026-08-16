from __future__ import annotations

from forge.nature_sim_v2 import NatureWorld, founder_genomes
from forge.nature_sim_v2.succession import choose_successor


def test_succession_prefers_direct_offspring_over_unrelated_survivor() -> None:
    world = NatureWorld(seed=991, size=40)
    genome = founder_genomes(variants_per_family=1)[0]
    parent_id = world.add_organism(genome, (10, 10), energy=.8)
    unrelated_id = world.add_organism(genome, (10.1, 10), energy=.8)
    child_id = world.add_organism(genome, (30, 30), energy=.8, parents=(parent_id,))
    parent = world.organisms[parent_id]
    parent.alive = False
    successor, message = choose_successor(world, parent)
    assert successor is not None and successor.entity_id == child_id
    assert "OFFSPRING" in message and world.events[-1]["type"] == "player_succession"
    assert unrelated_id != successor.entity_id


def test_succession_falls_back_to_any_living_body() -> None:
    world = NatureWorld(seed=992, size=40)
    genomes = founder_genomes(variants_per_family=1)
    dead_id = world.add_organism(genomes[0], (2, 2), energy=.8)
    survivor_id = world.add_organism(genomes[4], (36, 36), energy=.8)
    world.organisms[dead_id].alive = False
    successor, message = choose_successor(world, world.organisms[dead_id])
    assert successor is not None and successor.entity_id == survivor_id
    assert "SURVIVOR" in message
