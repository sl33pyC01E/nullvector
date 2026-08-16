from __future__ import annotations

from forge.creature_stage_developmental import develop
from forge.nature_sim_v2 import AdventureState,CreatureCreator,NatureWorld


def test_creator_generates_all_families_variants_and_valid_cellular_grafts() -> None:
    creator=CreatureCreator()
    hashes=set()
    for family in range(5):
        creator.family=family
        for variant in range(6):
            creator.variant=variant;genome=creator.genome(seed=1000+family*10+variant);organism=develop(genome.developmental);assert organism.cell_count>30;hashes.add(genome.semantic_sha256())
    assert len(hashes)==30
    creator.family=1;creator.donor_family=4;creator.graft_kind="locomotor";grafted=creator.genome(seed=9911);assert develop(grafted.developmental).cell_count>30 and any("graft" in value for value in grafted.mutation_log)


def test_creator_spends_resources_and_incarnates_playable_lineage() -> None:
    world=NatureWorld(seed=72,size=40);adventure=AdventureState(seed=73,size=40);adventure.inventory.update({"biomass":9,"knowledge":5,"metal":3,"crystal":3});creator=CreatureCreator(family=3,variant=2,donor_family=0,graft_kind="organ");creator.toggle_offer(0);creator.toggle_offer(2);before=dict(adventure.inventory);entity_id=creator.incarnate(world,adventure,(18,19),seed=12345);entity=world.organisms[entity_id]
    assert entity.genome.lineage_id.startswith("created-anomaly") and entity.body.organism.cell_count>30
    assert adventure.inventory["biomass"]<before["biomass"] and adventure.inventory["knowledge"]<before["knowledge"] and adventure.inventory["crystal"]<before["crystal"]
    assert world.events[-1]["type"]=="incarnation"


def test_creator_rejects_incarnation_without_physical_resources() -> None:
    world=NatureWorld(seed=74,size=32);adventure=AdventureState(seed=75,size=32);creator=CreatureCreator()
    try:creator.incarnate(world,adventure,(4,4),seed=9)
    except ValueError as exc:assert "needs" in str(exc)
    else:raise AssertionError("resource-free incarnation was accepted")
