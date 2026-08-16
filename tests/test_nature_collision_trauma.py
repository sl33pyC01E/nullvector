from __future__ import annotations

import numpy as np

from forge.nature_sim_v2 import NatureWorld,founder_genomes


def test_hostile_high_speed_collision_transfers_momentum_and_damages_cells() -> None:
    world=NatureWorld(seed=4,size=32);genomes=founder_genomes(variants_per_family=1);left=world.add_organism(genomes[0],(12,12),energy=.8);right=world.add_organism(genomes[1],(12.5,12),energy=.8);a,b=world.organisms[left],world.organisms[right];a.velocity[:]=(2.2,0);b.velocity[:]=(-2.2,0);before=(a.body.health.sum(),b.body.health.sum());world._resolve_collisions();after=(a.body.health.sum(),b.body.health.sum());assert after[0]<before[0] and after[1]<before[1];assert any(event["type"]=="collision_trauma" for event in world.events);assert a.velocity[0]<2.2 and b.velocity[0]>-2.2


def test_kin_overlap_without_collision_damage_for_reproduction() -> None:
    world=NatureWorld(seed=6,size=32);genome=founder_genomes(variants_per_family=1)[0];left=world.add_organism(genome,(10,10),energy=.8);right=world.add_organism(genome,(10.1,10),energy=.8);world.organisms[left].velocity[:]=(3,0);world.organisms[right].velocity[:]=(-3,0);health=[world.organisms[i].body.health.copy() for i in (left,right)];positions=[world.organisms[i].position.copy() for i in (left,right)];world._resolve_collisions();assert all(np.array_equal(world.organisms[i].body.health,value) for i,value in zip((left,right),health));assert all(np.array_equal(world.organisms[i].position,value) for i,value in zip((left,right),positions))
