from __future__ import annotations

from types import SimpleNamespace

from forge.nature_sim_v2 import AdventureState,NatureWorld,founder_genomes
from forge.qud_quests_v1 import QuestJournal
from forge.qud_services_v1 import use_settlement_service


def test_clinic_consumes_finite_medicine_and_repairs_actual_body() -> None:
    world=NatureWorld(seed=2401,size=40);entity=world.organisms[world.add_organism(founder_genomes(variants_per_family=1)[0],(20,20),energy=.8)];entity.body.impact((0,0),5,.5);before=float(entity.body.health.mean());settlement=SimpleNamespace(buildings=[SimpleNamespace(purpose="clinic")],stockpiles={"medicine":.2},wealth=.1);faction=SimpleNamespace(faction_id="f-test",name="Test Kin");adventure=AdventureState(seed=2402,size=40);journal=QuestJournal();message=use_settlement_service(settlement,faction,entity=entity,adventure=adventure,journal=journal)
    assert "CLINIC" in message
    assert settlement.stockpiles["medicine"]<.2
    assert float(entity.body.health.mean())>before
    assert journal.reputation["f-test"]>0


def test_observatory_transfers_finite_knowledge() -> None:
    world=NatureWorld(seed=2403,size=40);entity=world.organisms[world.add_organism(founder_genomes(variants_per_family=1)[3],(20,20),energy=1)];settlement=SimpleNamespace(buildings=[SimpleNamespace(purpose="observatory")],stockpiles={"knowledge":.2},wealth=.1);faction=SimpleNamespace(faction_id="f-phase",name="Phase Choir");adventure=AdventureState(seed=2404,size=40);journal=QuestJournal();message=use_settlement_service(settlement,faction,entity=entity,adventure=adventure,journal=journal)
    assert "OBSERVATORY" in message and adventure.inventory["knowledge"]>0
    assert settlement.stockpiles["knowledge"]<.2
