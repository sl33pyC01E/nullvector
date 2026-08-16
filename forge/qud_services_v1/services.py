from __future__ import annotations


def use_settlement_service(settlement, faction, *, entity, adventure, journal) -> str:
    purposes={building.purpose for building in settlement.buildings};stock=settlement.stockpiles;systems=entity.body.systems();service=None
    if "clinic" in purposes and stock.get("medicine",0)>=.04 and (systems["integrity"]<.98 or systems["circulation"]<.98):
        stock["medicine"]-=.04;entity.body.energy=min(1.2,entity.body.energy+.05);entity.body.heal((0,0),16,.16);entity.energy=min(1.2,entity.energy+.05);service="CLINIC // CELL REPAIR + SCAR MAPPING"
    elif entity.family==4 and "battery_hall" in purposes and stock.get("energy",0)>=.05 and entity.energy<1:
        transfer=min(.16,1.1-entity.energy,float(stock["energy"]));stock["energy"]-=transfer;entity.energy+=transfer;entity.body.energy=min(1.2,entity.body.energy+transfer);service=f"BATTERY HALL // +{transfer:.2f} CHARGE"
    elif "granary" in purposes and stock.get("food",0)>=.06 and entity.energy<1:
        transfer=min(.14,1.05-entity.energy,float(stock["food"])*.72);stock["food"]-=transfer/.72;entity.energy+=transfer;entity.reserve=min(1,entity.reserve+transfer*.45);service=f"GRANARY // +{transfer:.2f} METABOLIC ENERGY"
    elif "observatory" in purposes and stock.get("knowledge",0)>=.03:
        transfer=min(.08,float(stock["knowledge"]));stock["knowledge"]-=transfer;adventure.inventory["knowledge"]+=transfer;service=f"OBSERVATORY // +{transfer:.2f} KNOWLEDGE"
    elif "graft_house" in purposes and stock.get("parts",0)>=.04 and systems["locomotion"]<.98:
        stock["parts"]-=.04;entity.body.energy=min(1.2,entity.body.energy+.04);entity.body.heal((0,8),12,.12);service="GRAFT HOUSE // LOCOMOTOR TISSUE STABILIZED"
    if service is None:return "SETTLEMENT SERVICE // NO APPLICABLE STOCK OR FACILITY"
    journal.reputation[faction.faction_id]=min(1,journal.reputation.get(faction.faction_id,0)+.006);settlement.wealth+=.004;adventure.score+=2;return f"{faction.name.upper()} // {service}"
