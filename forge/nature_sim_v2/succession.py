from __future__ import annotations

import numpy as np


def choose_successor(world, previous) -> tuple[object | None, str]:
    living = [entity for entity in world.organisms.values() if entity.alive and entity.entity_id != previous.entity_id]
    if not living:
        return None, "EXTINCTION // NO LIVING BODY REMAINS"
    previous_parents = set(previous.parent_ids)

    def score(candidate) -> tuple[float, int]:
        relationship = 0.0
        if previous.entity_id in candidate.parent_ids:
            relationship += 120.0
        if candidate.entity_id in previous_parents:
            relationship += 95.0
        shared_parents = len(previous_parents.intersection(candidate.parent_ids))
        relationship += shared_parents * 72.0
        if candidate.genome.lineage_id == previous.genome.lineage_id:
            relationship += 48.0
        if previous.colony_id is not None and candidate.colony_id == previous.colony_id:
            relationship += 32.0
        if candidate.family == previous.family:
            relationship += 14.0
        distance = float(np.linalg.norm(world._delta(previous.position, candidate.position)))
        health = float(candidate.body.systems()["neural"] + candidate.body.systems()["integrity"])
        return relationship + health * 5.0 - distance * .42, -candidate.entity_id

    successor = max(living, key=score)
    relation = "offspring" if previous.entity_id in successor.parent_ids else "parent" if successor.entity_id in previous_parents else "sibling" if previous_parents.intersection(successor.parent_ids) else "lineage" if successor.genome.lineage_id == previous.genome.lineage_id else "colony" if previous.colony_id is not None and successor.colony_id == previous.colony_id else "survivor"
    world.events.append({"tick": world.tick_index, "type": "player_succession", "from": previous.entity_id, "to": successor.entity_id, "relation": relation})
    return successor, f"SUCCESSION // {relation.upper()} {successor.entity_id} // GENERATION {successor.genome.developmental.generation}"
