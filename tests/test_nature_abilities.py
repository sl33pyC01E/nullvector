from __future__ import annotations

import numpy as np

from forge.nature_sim_v2 import NatureWorld,entity_abilities,founder_genomes,use_ability


def test_five_families_have_distinct_anatomical_ability_sets() -> None:
    world=NatureWorld(seed=2,size=48);sets=[]
    for index,genome in enumerate(founder_genomes(variants_per_family=1)):entity_id=world.add_organism(genome,(10+index*3,10),energy=.9);sets.append(tuple(ability.ability_id for ability in entity_abilities(world.organisms[entity_id])))
    assert len(set(sets))==5;assert "spore" in sets[2] and "phase_fold" in sets[3] and {"bolt","beam"}<=set(sets[4])


def test_plant_spores_modify_real_ecology_and_anomaly_folds_position() -> None:
    world=NatureWorld(seed=8,size=48);genomes=founder_genomes(variants_per_family=1);plant=world.organisms[world.add_organism(genomes[2],(12,12),energy=.9)];anomaly=world.organisms[world.add_organism(genomes[3],(20,20),energy=.9)];flora=float(world.fields[8].sum());message=use_ability(world,plant,next(value for value in entity_abilities(plant) if value.ability_id=="spore"),(14,14));assert "SPORE" in message and world.fields[8].sum()>flora;before=anomaly.position.copy();message=use_ability(world,anomaly,next(value for value in entity_abilities(anomaly) if value.ability_id=="phase_fold"),(27,20));assert "PHASE FOLD" in message and not np.array_equal(before,anomaly.position)


def test_machine_bolt_enters_the_physical_projectile_world() -> None:
    world=NatureWorld(seed=9,size=48);machine=world.organisms[world.add_organism(founder_genomes(variants_per_family=1)[4],(10,10),energy=.9)];ability=next(value for value in entity_abilities(machine) if value.ability_id=="bolt");use_ability(world,machine,ability,(20,10));assert world.materials.projectiles and world.materials.projectiles[0].owner_id==machine.entity_id
