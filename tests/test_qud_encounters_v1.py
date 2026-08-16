from __future__ import annotations

from forge.nature_sim_v2 import AdventureState,NatureWorld,founder_genomes,load_session,save_session
from forge.nature_world_scale_v1 import InfiniteNatureAtlas,RegionKey
from forge.qud_encounters_v1 import generate_encounter,resolve_encounter
from forge.qud_quests_v1 import QuestJournal
from forge.qud_society_v1 import SocietyLayer


def test_encounters_are_deterministic_and_cover_all_site_kinds() -> None:
    kinds=AdventureState.SITE_KINDS;left=[generate_encounter(seed=7,site_id=f"s-{kind}",kind=kind) for kind in kinds];right=[generate_encounter(seed=7,site_id=f"s-{kind}",kind=kind) for kind in kinds];assert left==right and len({item.title for item in left})==len(kinds) and all(len(item.choices)==3 for item in left)


def test_resolution_modifies_physical_body_or_campaign_resources() -> None:
    world=NatureWorld(seed=71,size=40);entity_id=world.add_organism(founder_genomes(variants_per_family=1)[0],(18,18),energy=.8);entity=world.organisms[entity_id];adventure=AdventureState(seed=72,size=40);encounter=generate_encounter(seed=72,site_id="vault-x",kind="relic_vault");before_inventory=sum(adventure.inventory.values());before_health=entity.body.health.copy();message=resolve_encounter(encounter,1,world=world,entity=entity,adventure=adventure)
    assert encounter.resolved and message
    assert sum(adventure.inventory.values())>before_inventory or not (entity.body.health==before_health).all()
    assert world.events[-1]["type"]=="site_encounter"


def test_pending_encounter_survives_full_campaign_save(tmp_path) -> None:
    world=NatureWorld(seed=73,size=40);entity_id=world.add_organism(founder_genomes(variants_per_family=1)[0],(10,10),energy=.8);adventure=AdventureState(seed=74,size=40);site=adventure.sites[0];world.organisms[entity_id].position=site.position.copy();adventure.interact(world,world.organisms[entity_id]);assert adventure.pending_encounter
    society=SocietyLayer(world,seed=75);quests=QuestJournal();atlas=InfiniteNatureAtlas(seed=76);path=tmp_path/"encounter.nvs";save_session(world=world,adventure=adventure,society=society,quests=quests,atlas=atlas,region=RegionKey(0,0),selected=entity_id,path=path);restored=load_session(path);assert restored["adventure"].pending_encounter==adventure.pending_encounter;assert restored["adventure"].encounters[adventure.pending_encounter].title==adventure.encounters[adventure.pending_encounter].title


def test_successful_risky_site_choice_grows_physical_heritable_anatomy() -> None:
    world=NatureWorld(seed=77,size=40);entity=world.organisms[world.add_organism(founder_genomes(variants_per_family=1)[0],(20,20),energy=.9)];adventure=AdventureState(seed=78,size=40);before=len(entity.genome.developmental.components)
    for index in range(64):
        encounter=generate_encounter(seed=78,site_id=f"grove-{index}",kind="grove");message=resolve_encounter(encounter,1,world=world,entity=entity,adventure=adventure)
        if message.startswith("SUCCESS"):break
    assert "MUTATION STORAGE LOBES" in message
    assert len(entity.genome.developmental.components)==before+2
    assert entity.genome.developmental.generation==1
