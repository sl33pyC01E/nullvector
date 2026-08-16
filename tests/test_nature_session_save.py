from __future__ import annotations

import numpy as np

from forge.nature_sim_v2 import AdventureState, ColonyState, NatureWorld, load_session, save_session
from forge.nature_world_scale_v1 import InfiniteNatureAtlas, RegionKey
from forge.qud_quests_v1 import QuestJournal
from forge.qud_society_v1 import SocietyLayer


def test_campaign_save_restores_progress_society_contracts_and_atlas(tmp_path) -> None:
    world = NatureWorld(seed=7001, size=40)
    world.seed_founders(variants_per_family=3)
    humanoids = [entity for entity in world.organisms.values() if entity.family == 0][:3]
    members = {entity.entity_id for entity in humanoids}
    for entity in humanoids:
        entity.colony_id = 1
    world.colonies[1] = ColonyState(1, 0, humanoids[0].genome.lineage_id, members, np.asarray((12.0, 13.0)))
    world.next_colony_id = 2

    adventure = AdventureState(seed=8002, size=world.size)
    adventure.inventory.update({"rock": 4.0, "metal": 3.0, "crystal": 1.0, "biomass": 2.0})
    adventure.craft_selected()
    adventure.discoveries.add(adventure.sites[0].site_id)
    adventure.sites[0].discovered = True
    adventure.sites[0].richness = 0.25
    adventure.score = 91

    society = SocietyLayer(world, seed=9003)
    society.found_from_colony(1)
    society.step_history(2)
    journal = QuestJournal()
    journal.accept_nearest(society, world, humanoids[0], adventure)
    journal.reputation[next(iter(society.factions))] = 0.25

    atlas = InfiniteNatureAtlas(seed=10004)
    region = RegionKey(2, -3, 1)
    atlas.record(region, world)
    before = (world.snapshot().semantic_sha256, society.semantic_sha256(), atlas.semantic_sha256())

    path = tmp_path / "campaign.nvs"
    save_session(world=world, adventure=adventure, society=society, quests=journal, atlas=atlas, region=region, selected=humanoids[0].entity_id, path=path)
    restored = load_session(path)

    assert restored["world"].snapshot().semantic_sha256 == before[0]
    assert restored["society"].semantic_sha256() == before[1]
    assert restored["atlas"].semantic_sha256() == before[2]
    assert restored["region"] == region
    assert restored["selected"] == humanoids[0].entity_id
    assert restored["adventure"].inventory == adventure.inventory
    assert restored["adventure"].equipped == adventure.equipped
    assert restored["adventure"].sites[0].richness == 0.25
    assert restored["quests"].reputation == journal.reputation
    assert tuple(restored["quests"].entries) == tuple(journal.entries)


def test_campaign_save_rejects_missing_selected_entity(tmp_path) -> None:
    world = NatureWorld(seed=3, size=32)
    world.seed_founders(variants_per_family=1)
    adventure = AdventureState(seed=4, size=32)
    society = SocietyLayer(world, seed=5)
    journal = QuestJournal()
    atlas = InfiniteNatureAtlas(seed=6)
    path = tmp_path / "campaign.nvs"
    try:
        save_session(world=world, adventure=adventure, society=society, quests=journal, atlas=atlas, region=RegionKey(0, 0), selected=999999, path=path)
        load_session(path)
    except ValueError as exc:
        assert "selected entity" in str(exc)
    else:
        raise AssertionError("missing selected entity was accepted")
