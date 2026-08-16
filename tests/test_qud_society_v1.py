from __future__ import annotations

import numpy as np

from forge.nature_sim_v2 import ColonyState,NatureWorld,founder_genomes
from forge.qud_society_v1 import SocietyLayer,expand_settlement,generate_building,validate_building


def _society_world()->NatureWorld:
    world=NatureWorld(seed=23,size=64);genome=founder_genomes(variants_per_family=1)[0];ids=[world.add_organism(genome,(20+i*.3,20),energy=.8) for i in range(5)];world.colonies[1]=ColonyState(1,0,genome.lineage_id,set(ids),np.asarray((20.6,20.0)));[setattr(world.organisms[i],"colony_id",1) for i in ids];return world


def test_buildings_are_conglomerate_connected_and_deterministic() -> None:
    left=generate_building(seed=4,origin=(10,10));right=generate_building(seed=4,origin=(10,10))
    assert left==right and all(validate_building(left).values())
    buildings,roads=expand_settlement(seed=9,center=(32,32),count=12)
    assert len(buildings)==12 and len(roads)>20 and all(all(validate_building(b).values()) for b in buildings)


def test_colony_can_found_a_differentiated_society() -> None:
    layer=SocietyLayer(_society_world(),seed=71);faction_id=layer.found_from_colony(1);faction=layer.factions[faction_id];settlement=layer.settlements[next(iter(faction.settlement_ids))]
    assert faction.name and len(faction.cultural_traits)==8 and "organ_grafting" in faction.technologies
    assert len(settlement.buildings)>=3 and settlement.population==5


def test_history_activities_and_relations_replay_exactly() -> None:
    left=SocietyLayer(_society_world(),seed=88);right=SocietyLayer(_society_world(),seed=88)
    for layer in (left,right):layer.found_from_colony(1);layer.step_history(30)
    assert left.semantic_sha256()==right.semantic_sha256()
    assert left.history and left.activities and len(next(iter(left.factions.values())).technologies)>4

