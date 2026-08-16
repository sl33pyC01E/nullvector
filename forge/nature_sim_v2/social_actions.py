from __future__ import annotations

import numpy as np

from .state import ColonyState


def bond_nearby(world,entity)->str:
    if entity.colony_id in world.colonies:
        colony=world.colonies[entity.colony_id]
        return f"KIN BOND // ALREADY COLONY {colony.colony_id} // {len(colony.member_ids)} MEMBERS"
    radius=2.8+2.4*entity.genome.trait("sociability");candidates=[other for other in world.organisms.values() if other.alive and other.entity_id!=entity.entity_id and other.family==entity.family and np.linalg.norm(world._delta(entity.position,other.position))<=radius]
    if not candidates:return "KIN BOND // NO COMPATIBLE NEIGHBOR IN SENSORY RANGE"
    candidate=min(candidates,key=lambda other:(float(np.linalg.norm(world._delta(entity.position,other.position))),other.entity_id))
    if candidate.colony_id in world.colonies:
        colony=world.colonies[candidate.colony_id];colony.member_ids.add(entity.entity_id);entity.colony_id=colony.colony_id;kind="joined"
    else:
        colony_id=world.next_colony_id;world.next_colony_id+=1;members={entity.entity_id,candidate.entity_id};center=np.mean([world.organisms[item].position for item in sorted(members)],axis=0);colony=ColonyState(colony_id,entity.family,entity.genome.lineage_id,members,center);world.colonies[colony_id]=colony
        for member_id in members:world.organisms[member_id].colony_id=colony_id
        kind="founded"
    world.events.append({"tick":world.tick_index,"type":"kin_bond","entity":entity.entity_id,"partner":candidate.entity_id,"colony":colony.colony_id,"kind":kind});return f"KIN BOND // {kind.upper()} COLONY {colony.colony_id} // {len(colony.member_ids)} MEMBERS"
