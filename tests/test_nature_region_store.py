from __future__ import annotations

import numpy as np

from forge.nature_sim_v2 import ColonyState,NatureWorld, PersistentRegionStore,founder_genomes
from forge.nature_world_scale_v1 import RegionKey
from forge.qud_society_v1 import SocietyLayer


def test_region_store_preserves_ecology_without_duplicating_traveller(tmp_path) -> None:
    world=NatureWorld(seed=404,size=40);world.seed_founders(variants_per_family=2);traveller=world.organisms[min(world.organisms)];resident=world.organisms[sorted(world.organisms)[1]];resident.body.impact((0,0),5,.42);world.fields[8,4,7]=.12345;world.materials.deposit("crystal",(8,9),2.5,1.2)
    store=PersistentRegionStore(tmp_path,atlas_seed=123);key=RegionKey(-7,12,2);before_health=resident.body.health.copy();before_count=len(world.organisms);report=store.save(key,world,exclude_entity_id=traveller.entity_id)
    assert store.exists(key) and len(world.organisms)==before_count and traveller.entity_id in world.organisms and report["excluded"]==traveller.entity_id
    restored=store.load(key);assert restored is not None and len(restored.organisms)==before_count-1 and traveller.entity_id not in restored.organisms
    assert np.array_equal(restored.organisms[resident.entity_id].body.health,before_health)
    assert restored.fields[8,4,7]==.12345 and restored.materials.semantic_sha256()==world.materials.semantic_sha256()


def test_region_store_returns_none_for_undiscovered_region(tmp_path) -> None:
    store=PersistentRegionStore(tmp_path,atlas_seed=999)
    assert store.load(RegionKey(1,2)) is None


def test_region_store_preserves_factions_cities_stockpiles_and_history(tmp_path) -> None:
    world=NatureWorld(seed=405,size=40);genome=next(item for item in founder_genomes(variants_per_family=1) if item.family==0);ids=[world.add_organism(genome,(18+index*.2,18),energy=.8) for index in range(4)];world.colonies[1]=ColonyState(1,0,genome.lineage_id,set(ids),np.asarray((18.3,18.0)))
    for entity_id in ids:world.organisms[entity_id].colony_id=1
    society=SocietyLayer(world,seed=406);faction_id=society.found_from_colony(1);society.step_history(2);settlement=society.settlements[next(iter(society.factions[faction_id].settlement_ids))];settlement.stockpiles["crystal"]=1.234
    store=PersistentRegionStore(tmp_path,atlas_seed=124);key=RegionKey(3,-8,0);store.save(key,world,society=society);restored_world,restored_society=store.load(key,include_society=True)
    assert restored_society is not None and restored_society.nature is restored_world
    assert restored_society.factions[faction_id].technologies==society.factions[faction_id].technologies
    assert restored_society.settlements[settlement.settlement_id].stockpiles["crystal"]==1.234
    assert len(restored_society.history)==len(society.history)
