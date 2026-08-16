from __future__ import annotations

import numpy as np

from forge.nature_sim_v2 import ColonyState,NatureWorld,founder_genomes


class _Roles:
    def __init__(self,role:str):self.role=role
    def assign(self,members,state):return {member.entity_id:self.role for member in members}
    def action(self,entity_id):return (.1,.1,.1)


def _colony(role:str):
    world=NatureWorld(seed=1241,size=40);genome=founder_genomes(variants_per_family=1)[0];ids=[world.add_organism(genome,(20+index*.1,20),energy=.96) for index in range(4)];world.colonies[1]=ColonyState(1,0,genome.lineage_id,set(ids),np.asarray((20.,20.)));world.colony_ecology.role_policy=_Roles(role)
    for entity_id in ids:world.organisms[entity_id].colony_id=1
    return world


def test_gatherers_remove_finite_local_fields_into_colony_store() -> None:
    world=_colony("gatherer");world.fields[9,20,20]=1;before=float(world.fields[:,20,20].sum());world.colony_ecology.step(world,.5);state=world.colony_ecology.states[1]
    assert float(world.fields[:,20,20].sum())<before
    assert float(state.material_store.sum())>0


def test_builders_consolidate_store_into_visible_powder_cache() -> None:
    world=_colony("builder");world.tick_index=1;world.colony_ecology.step(world,.2);state=world.colony_ecology.states[1];state.material_store[9]=.2;world.tick_index=91;before=float(world.materials.mass.sum());world.colony_ecology.step(world,.2)
    assert float(world.materials.mass.sum())>before
    assert world.events[-1]["type"]=="colony_cache"
