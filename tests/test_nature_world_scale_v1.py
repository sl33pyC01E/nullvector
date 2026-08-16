from forge.nature_world_scale_v1 import InfiniteNatureAtlas,RegionKey
from forge.nature_sim_v2 import NatureWorld


def test_unbounded_region_descriptions_are_deterministic_and_distinct() -> None:
    atlas=InfiniteNatureAtlas(seed=7);keys=(RegionKey(0,0),RegionKey(2**31,-2**31),RegionKey(-9,14,3));first=[atlas.describe(key) for key in keys];second=[atlas.describe(key) for key in keys]
    assert first==second and len({item.seed for item in first})==3 and len(atlas.visited)==0
    assert len(atlas.window(keys[0],2))==25


def test_visited_region_records_exact_ecology_departure() -> None:
    atlas=InfiniteNatureAtlas(seed=8);key=RegionKey(4,-3);world=NatureWorld(seed=atlas.region_seed(key),size=32,max_population=32);world.seed_founders(variants_per_family=1);summary=atlas.record(key,world)
    assert summary.world_sha256==world.snapshot().semantic_sha256 and summary.population==(1,1,1,1,1) and summary.visits==1
    assert atlas.describe(key)==summary and len(atlas.semantic_sha256())==64
