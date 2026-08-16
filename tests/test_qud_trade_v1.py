from __future__ import annotations

import numpy as np

from forge.nature_sim_v2 import AdventureState, ColonyState, NatureWorld, founder_genomes
from forge.qud_quests_v1 import QuestJournal
from forge.qud_society_v1 import SocietyLayer
from forge.qud_trade_v1 import execute_trade, generate_trade_offers


def _settlement():
    world = NatureWorld(seed=801, size=40)
    ids = [world.add_organism(founder_genomes(variants_per_family=1)[0], (18 + index * .2, 18), energy=.8) for index in range(3)]
    genome = world.organisms[ids[0]].genome
    colony = ColonyState(1, 0, genome.lineage_id, set(ids), np.asarray((18.2, 18.0)))
    world.colonies[1] = colony
    for entity_id in ids:
        world.organisms[entity_id].colony_id = 1
    society = SocietyLayer(world, seed=802)
    faction_id = society.found_from_colony(colony.colony_id)
    return world, society, society.settlements[next(iter(society.factions[faction_id].settlement_ids))]


def test_trade_offers_are_deterministic_and_stock_aware() -> None:
    _, _, settlement = _settlement()
    settlement.stockpiles.update({"food": 5.0, "water": 4.0, "mineral": .01, "parts": .02, "crystal": .03, "knowledge": .04})
    left = generate_trade_offers(settlement, reputation=.2, epoch=4)
    right = generate_trade_offers(settlement, reputation=.2, epoch=4)
    assert left == right and len(left) == 3
    assert {offer.give_material for offer in left} <= {"rock", "metal", "crystal", "knowledge"}


def test_trade_moves_finite_stock_both_directions() -> None:
    world, _, settlement = _settlement()
    settlement.stockpiles.update({"food": 4.0, "water": 3.0, "mineral": .01, "parts": .02, "crystal": .03, "knowledge": .04})
    adventure = AdventureState(seed=803, size=world.size)
    adventure.inventory.update({name: 3.0 for name in adventure.inventory})
    journal = QuestJournal()
    offer = generate_trade_offers(settlement, reputation=0, epoch=0)[0]
    before_player = dict(adventure.inventory); before_settlement = dict(settlement.stockpiles)
    message = execute_trade(offer, settlement=settlement, adventure=adventure, journal=journal)
    assert message.startswith("BARTERED")
    assert adventure.inventory[offer.give_material] < before_player[offer.give_material]
    assert adventure.inventory[offer.receive_material] > before_player[offer.receive_material]
    assert settlement.stockpiles[PLAYER_TO_SETTLEMENT_FOR_TEST[offer.give_material]] > before_settlement[PLAYER_TO_SETTLEMENT_FOR_TEST[offer.give_material]]


PLAYER_TO_SETTLEMENT_FOR_TEST = {"biomass":"food", "rock":"mineral", "metal":"parts", "crystal":"crystal", "water":"water", "knowledge":"knowledge"}
