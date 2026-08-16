from __future__ import annotations

import numpy as np

from forge.nature_sim_v2 import NatureWorld
from forge.nature_world_scale_v1 import BIOMES,InfiniteNatureAtlas,RegionKey


def test_biomes_materially_change_ecology_and_make_organized_ruins() -> None:
    atlas=InfiniteNatureAtlas(seed=808);found={}
    for y in range(-12,13):
        for x in range(-12,13):
            key=RegionKey(x,y);summary=atlas.describe(key)
            if summary.biome not in found:found[summary.biome]=key
    assert set(found)==set(BIOMES)
    hashes=[]
    for biome,key in sorted(found.items()):
        world=NatureWorld(seed=atlas.region_seed(key),size=48);before=world.fields.copy();summary=atlas.terraform(world,key);assert summary.biome==biome and not np.array_equal(before,world.fields);assert np.count_nonzero(world.materials.structure_id)>10;hashes.append((world.fields.tobytes(),world.materials.semantic_sha256()))
    assert len({material for _,material in hashes})==len(BIOMES)
