from __future__ import annotations

import numpy as np

from forge.nature_sim_v2 import ColonyState,NatureWorld,founder_genomes,load_world,save_world


def _colony_world(count=12):
    world=NatureWorld(seed=909,size=48,max_population=80);genome=founder_genomes(variants_per_family=1)[2];ids=[world.add_organism(genome,(20+(index%4)*.25,20+(index//4)*.25),energy=.5+.02*(index%3)) for index in range(count)];world.colonies[1]=ColonyState(1,2,genome.lineage_id,set(ids),np.asarray((20.,20.)));world.next_colony_id=2
    for entity_id in ids:world.organisms[entity_id].colony_id=1
    return world


def test_colony_culture_adapts_and_changes_reproduction_timing() -> None:
    world=_colony_world(8);state=world.colony_culture.ensure(world,1);before=state.values.copy();world.colony_ecology.step(world,.2);ecology=world.colony_ecology.states[1];breeder=next(entity_id for entity_id,role in ecology.assignments.items() if role=="breeder");world.organisms[breeder].reproduction_cooldown=10;before_cooldown=world.organisms[breeder].reproduction_cooldown
    for _ in range(20):world.colony_culture.step(world,.2)
    assert not np.array_equal(state.values,before) and state.observations==20
    assert world.organisms[breeder].reproduction_cooldown<before_cooldown


def test_colony_fission_inherits_culture_with_bounded_variation() -> None:
    world=_colony_world(18);parent=world.colony_culture.ensure(world,1);world._update_colonies();child_ids=[value for value in world.colonies if value!=1];assert child_ids;child=world.colony_culture.ensure(world,child_ids[0]);assert child.parent_colony==1 and child.generation==parent.generation+1;assert np.max(np.abs(child.values-parent.values))<=.041


def test_colony_culture_survives_save(tmp_path) -> None:
    world=_colony_world(8);world.colony_culture.step(world,.2);path=tmp_path/"culture.nvz";save_world(world,path);restored=load_world(path);left=world.colony_culture.payload();right=restored.colony_culture.payload();assert left==right
