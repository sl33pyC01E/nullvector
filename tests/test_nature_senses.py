from __future__ import annotations

from forge.nature_sim_v2 import NatureWorld,founder_genomes,sensory_field,visible_targets


def test_family_senses_have_direction_range_and_radial_modes() -> None:
    world=NatureWorld(seed=14,size=48);ids=[]
    for family,genome in enumerate(founder_genomes(variants_per_family=1)):ids.append(world.add_organism(genome,(20+family*.4,20),energy=.8))
    fields=[sensory_field(world.organisms[entity_id]) for entity_id in ids];assert all(field.range>2 and field.arc_radians>0 for field in fields);assert fields[2].radial and fields[3].radial and not fields[0].radial;assert visible_targets(world,world.organisms[ids[3]],fields[3])


def test_destroying_sensor_organs_reduces_live_range() -> None:
    world=NatureWorld(seed=17,size=32);entity_id=world.add_organism(founder_genomes(variants_per_family=1)[0],(12,12),energy=.8);entity=world.organisms[entity_id];before=sensory_field(entity).range
    sensors=[component for component in entity.genome.developmental.components if component.organ=="sensor" or component.kind in ("head","sensor_crown")]
    for component in sensors:entity.body.impact(component.anchor,max(component.radius)*1.4,1)
    assert sensory_field(entity).range<before
