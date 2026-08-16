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


def test_settlement_walls_become_destructible_powder_structures() -> None:
    world=_society_world();layer=SocietyLayer(world,seed=77);layer.found_from_colony(1)
    assert layer.materialized_buildings and np.any(world.materials.structure_id>0)
    before=int(np.count_nonzero(world.materials.structure_id));wall=np.argwhere(world.materials.structure_id>0)[0]
    for _ in range(16):world.materials.beam((wall[1]-2,wall[0]),(wall[1]+2,wall[0]),energy=20,width=.5)
    assert int(np.count_nonzero(world.materials.structure_id))<before
    layer.step_history(1);assert layer.assignments


def test_settlement_economy_harvests_finite_fields_and_runs_building_chains() -> None:
    world=_society_world();layer=SocietyLayer(world,seed=177);faction_id=layer.found_from_colony(1);settlement=layer.settlements[next(iter(layer.factions[faction_id].settlement_ids))]
    before=float(world.fields[[0,2,3,4,8,9]].sum());layer.step_history(4);after=float(world.fields[[0,2,3,4,8,9]].sum())
    assert after<before
    assert settlement.production and settlement.stockpiles["food"]>=0
    assert settlement.stockpiles["medicine"]>0
    assert settlement.stockpiles["parts"]>0
    assert settlement.power>=0 and settlement.population==5


def test_material_surplus_constructs_a_new_physical_building() -> None:
    world=_society_world();layer=SocietyLayer(world,seed=277);faction_id=layer.found_from_colony(1);settlement=layer.settlements[next(iter(layer.factions[faction_id].settlement_ids))]
    before_buildings=len(settlement.buildings);before_cells=int(np.count_nonzero(world.materials.structure_id));settlement.stockpiles["mineral"]=4;settlement.stockpiles["parts"]=2;layer.step_history(1)
    assert len(settlement.buildings)==before_buildings+1 and settlement.projects_completed==1
    assert int(np.count_nonzero(world.materials.structure_id))>=before_cells
    assert any(event.kind=="construction" for event in layer.history)
