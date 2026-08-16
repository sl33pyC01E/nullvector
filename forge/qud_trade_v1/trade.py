from __future__ import annotations

from dataclasses import dataclass
import hashlib


PLAYER_TO_SETTLEMENT = {
    "biomass": "food",
    "rock": "mineral",
    "metal": "parts",
    "crystal": "crystal",
    "water": "water",
    "knowledge": "knowledge",
}
SETTLEMENT_TO_PLAYER = {value: key for key, value in PLAYER_TO_SETTLEMENT.items()}


@dataclass(frozen=True, slots=True)
class TradeOffer:
    offer_id: str
    settlement_id: str
    give_material: str
    give_amount: float
    receive_material: str
    receive_amount: float
    relation: float


def generate_trade_offers(settlement, *, reputation: float, epoch: int) -> tuple[TradeOffer, ...]:
    stock = settlement.stockpiles
    commodities = tuple(SETTLEMENT_TO_PLAYER)
    scarce = sorted(commodities, key=lambda name: (stock.get(name, 0.0), name))
    surplus = sorted(commodities, key=lambda name: (-stock.get(name, 0.0), name))
    offers = []
    relation = max(-1.0, min(1.0, float(reputation)))
    generosity = 0.72 + 0.28 * (relation + 1.0) * 0.5
    for index in range(3):
        ask = scarce[index % len(scarce)]
        give = next(name for name in surplus[index:] + surplus[:index] if name != ask and stock.get(name, 0.0) > 0.04)
        digest = hashlib.sha256(f"{settlement.settlement_id}:{epoch}:{index}:{ask}:{give}".encode()).hexdigest()
        ask_amount = 0.18 + 0.06 * index
        receive_amount = min(float(stock.get(give, 0.0)) * 0.28, ask_amount * generosity * (1.0 + min(1.0, float(settlement.wealth)) * 0.12))
        offers.append(TradeOffer("trade-" + digest[:14], settlement.settlement_id, SETTLEMENT_TO_PLAYER[ask], ask_amount, SETTLEMENT_TO_PLAYER[give], max(0.02, receive_amount), relation))
    return tuple(offers)


def execute_trade(offer: TradeOffer, *, settlement, adventure, journal) -> str:
    if offer.settlement_id != settlement.settlement_id:
        raise ValueError("trade settlement drifted")
    if adventure.inventory.get(offer.give_material, 0.0) + 1e-9 < offer.give_amount:
        raise ValueError(f"needs {offer.give_amount:.2f} {offer.give_material}")
    settlement_receive = PLAYER_TO_SETTLEMENT[offer.receive_material]
    available = float(settlement.stockpiles.get(settlement_receive, 0.0))
    if available + 1e-9 < offer.receive_amount:
        raise ValueError("settlement stock changed")
    settlement_give = PLAYER_TO_SETTLEMENT[offer.give_material]
    adventure.inventory[offer.give_material] -= offer.give_amount
    adventure.inventory[offer.receive_material] += offer.receive_amount
    settlement.stockpiles[settlement_give] = settlement.stockpiles.get(settlement_give, 0.0) + offer.give_amount
    settlement.stockpiles[settlement_receive] = available - offer.receive_amount
    journal.reputation[settlement.faction_id] = min(1.0, journal.reputation.get(settlement.faction_id, 0.0) + 0.012)
    settlement.wealth += offer.give_amount * 0.025
    adventure.score += 3
    return f"BARTERED {offer.give_amount:.2f} {offer.give_material.upper()} // RECEIVED {offer.receive_amount:.2f} {offer.receive_material.upper()}"
