from __future__ import annotations

import numpy as np

from forge.nature_sim_v2 import ColonyState,NatureWorld,founder_genomes,load_world,save_world


def test_living_world_save_restores_exact_semantic_state_and_continuation(tmp_path) -> None:
    world=NatureWorld(seed=414,size=40);world.seed_founders(variants_per_family=1);ids=sorted(world.organisms);world.colonies[1]=ColonyState(1,0,world.organisms[ids[0]].genome.lineage_id,set(ids[:2]),np.asarray((12.,12.)))
    for entity_id in ids[:2]:world.organisms[entity_id].colony_id=1
    world.organisms[ids[0]].body.impact((0,0),4,.55);world.fire_projectile(ids[0],(20,20));
    for _ in range(12):world.step(.1,publish=False)
    before=world.snapshot();path=tmp_path/"living_world.nvz";report=save_world(world,path);restored=load_world(path);after=restored.snapshot();assert report["world_sha256"]==before.semantic_sha256==after.semantic_sha256;assert restored.events==world.events;assert restored.rng.bit_generator.state==world.rng.bit_generator.state
    world.step(.1,publish=False);restored.step(.1,publish=False);assert world.snapshot().semantic_sha256==restored.snapshot().semantic_sha256
