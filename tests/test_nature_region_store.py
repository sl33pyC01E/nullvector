from __future__ import annotations

import numpy as np

from forge.nature_sim_v2 import NatureWorld, PersistentRegionStore
from forge.nature_world_scale_v1 import RegionKey


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
