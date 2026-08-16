from __future__ import annotations

import numpy as np

from forge.creature_stage_developmental import develop
from forge.nature_sim_v2 import NatureWorld, founder_genomes, recombine


def test_founders_cover_families_and_spawn_without_cascade() -> None:
    founders=founder_genomes(variants_per_family=2)
    assert len(founders)==10
    assert [sum(g.family==family for g in founders) for family in range(5)]==[2]*5
    for genome in founders:
        organism=develop(genome.developmental)
        assert organism.cell_count>30
        assert len(organism.genome.components)>=4


def test_structural_recombination_changes_body_and_preserves_pairs() -> None:
    parents=founder_genomes(variants_per_family=2)
    child=recombine(parents[2],parents[3],seed=0xBEEFBEEF)
    assert child.developmental.generation==1
    assert len(child.developmental.parent_ids)==2
    assert child.semantic_sha256() not in {parents[2].semantic_sha256(),parents[3].semantic_sha256()}
    lookup={a.appendage_id:a for a in child.developmental.appendages}
    for appendage in lookup.values():
        if appendage.paired_with is not None:
            assert lookup[appendage.paired_with].paired_with==appendage.appendage_id
    assert develop(child.developmental).cell_count>30


def test_material_consumption_is_spatial_and_family_specific() -> None:
    world=NatureWorld(seed=7,size=32)
    founders=founder_genomes(variants_per_family=1)
    ids=[world.add_organism(g,(8+family*3,8),energy=.25) for family,g in enumerate(founders)]
    before=world.fields.copy()
    for _ in range(8):world.step(.25)
    assert np.any(world.fields<before)
    assert world.organisms[ids[2]].consumed[1]>0  # plant light
    assert world.organisms[ids[3]].consumed[4]>0  # anomaly phase
    assert world.organisms[ids[4]].consumed[3]>0  # machine charge


def test_world_replay_and_colonies_are_deterministic() -> None:
    left=NatureWorld(seed=91,size=32);right=NatureWorld(seed=91,size=32)
    for world in (left,right):world.seed_founders(variants_per_family=2)
    for _ in range(180):
        a=left.step(.25);b=right.step(.25)
        assert a.semantic_sha256==b.semantic_sha256
    assert left.snapshot()==right.snapshot()
    assert left.snapshot().population>0
    assert left.snapshot().colony_count>0


def test_reproduction_is_delayed_and_creates_developed_offspring() -> None:
    world=NatureWorld(seed=123,size=32,max_population=80)
    founders=founder_genomes(variants_per_family=1)
    # Put a compatible animal pair together and advance them to maturity.
    first=world.add_organism(founders[1],(12,12),energy=1.0)
    second=world.add_organism(founders[1],(12.5,12),energy=1.0)
    world.organisms[first].age=70;world.organisms[second].age=70
    for _ in range(200):
        world.step(.25)
        if world.births:break
    assert world.births>0
    children=[o for o in world.organisms.values() if o.parent_ids]
    assert children and children[0].genome.developmental.generation==1
    assert children[0].body.snapshot().systems["integrity"]==1


def test_death_decomposes_gradually_without_explosion() -> None:
    world=NatureWorld(seed=3,size=32)
    entity_id=world.add_organism(founder_genomes(variants_per_family=1)[1],(10,10),energy=.5)
    entity=world.organisms[entity_id]
    entity.body.health[:]=0
    world.step(.25)
    assert not entity.alive and entity.stage=="dead"
    assert entity_id in world.organisms
    start=entity.decomposition
    for _ in range(20):world.step(.25)
    assert entity.decomposition>start and entity.decomposition<1
