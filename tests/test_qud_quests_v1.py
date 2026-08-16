from __future__ import annotations

import numpy as np

from forge.nature_sim_v2 import AdventureState,ColonyState,NatureWorld,founder_genomes
from forge.qud_quests_v1 import QuestJournal
from forge.qud_society_v1 import SocietyLayer


def _world_and_society():
    world=NatureWorld(seed=71,size=48);genome=founder_genomes(variants_per_family=1)[0];ids=[world.add_organism(genome,(20+i*.2,20),energy=.8) for i in range(4)];world.colonies[1]=ColonyState(1,0,genome.lineage_id,set(ids),np.asarray((20.3,20.0)))
    for entity_id in ids:world.organisms[entity_id].colony_id=1
    society=SocietyLayer(world,seed=8);society.found_from_colony(1);society.step_history(1);return world,society,ids[0]


def test_society_contracts_bind_to_real_world_metrics_and_pay_rewards() -> None:
    world,society,entity_id=_world_and_society();adventure=AdventureState(seed=9,size=48);journal=QuestJournal();activity=next(iter(society.activities.values()));entry=journal.accept(activity,journal.metrics(world,adventure));assert entry.metric
    if entry.metric=="score":adventure.score+=entry.target
    elif entry.metric=="crafts":adventure.craft_count+=int(entry.target)
    elif entry.metric=="buildings":adventure.buildings.extend([object()]*int(entry.target))
    elif entry.metric=="grafts":world.events.extend({"type":"graft"} for _ in range(int(entry.target)))
    elif entry.metric=="births":world.births+=int(entry.target)
    elif entry.metric=="colonies":world.colonies[2]=world.colonies[1]
    elif entry.metric=="predations":world.predation_events+=int(entry.target)
    elif entry.metric=="artifacts":adventure.artifacts.extend([object()]*int(entry.target))
    else:adventure.discoveries.update(f"site-{i}" for i in range(int(entry.target)))
    before=adventure.score;finished=journal.observe(world,adventure);assert finished and entry.complete and adventure.score==before+75 and journal.reputation[entry.issuer]>.0


def test_nearest_contract_acceptance_is_idempotent() -> None:
    world,society,entity_id=_world_and_society();adventure=AdventureState(seed=3,size=48);journal=QuestJournal();message=journal.accept_nearest(society,world,world.organisms[entity_id],adventure);assert "ACCEPTED" in message and len(journal.entries)==1
