from __future__ import annotations

import hashlib
from pathlib import Path

from ..nature_world_scale_v1 import RegionKey
from .savegame import load_world, save_world


class PersistentRegionStore:
    """Additive on-disk region persistence for an effectively unbounded world."""

    def __init__(self, root: Path, *, atlas_seed: int) -> None:
        self.root = Path(root).resolve() / f"atlas-{int(atlas_seed):016x}"
        self.atlas_seed = int(atlas_seed)

    @staticmethod
    def _name(key: RegionKey) -> str:
        digest = hashlib.sha256(f"{key.x}:{key.y}:{key.depth}".encode()).hexdigest()[:12]
        return f"d{key.depth:+05d}_x{key.x:+09d}_y{key.y:+09d}_{digest}.nvz"

    def path(self, key: RegionKey) -> Path:
        return self.root / self._name(key)

    def exists(self, key: RegionKey) -> bool:
        return self.path(key).is_file()

    def save(self, key: RegionKey, world, *, exclude_entity_id: int | None = None) -> dict:
        """Save a region without duplicating the travelling player inside it."""
        removed = None
        colony_memberships: list[int] = []
        if exclude_entity_id is not None:
            removed = world.organisms.pop(int(exclude_entity_id), None)
            if removed is None:
                raise ValueError("region store excluded entity is missing")
            for colony_id, colony in world.colonies.items():
                if exclude_entity_id in colony.member_ids:
                    colony.member_ids.remove(exclude_entity_id)
                    colony_memberships.append(colony_id)
        try:
            report = save_world(world, self.path(key))
        finally:
            if removed is not None:
                world.organisms[removed.entity_id] = removed
                for colony_id in colony_memberships:
                    world.colonies[colony_id].member_ids.add(removed.entity_id)
        return {**report, "region": (key.x, key.y, key.depth), "excluded": exclude_entity_id}

    def load(self, key: RegionKey, *, motion_policy=None, behavior_policy=None, colony_policy=None):
        path = self.path(key)
        if not path.is_file():
            return None
        return load_world(path, motion_policy=motion_policy, behavior_policy=behavior_policy, colony_policy=colony_policy)
