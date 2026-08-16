from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import uuid

from ..nature_world_scale_v1 import RegionKey
from .savegame import load_world, save_world
from .session_save import _adventure_payload,_restore_adventure,_restore_society,_society_payload


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

    def civilization_path(self,key:RegionKey)->Path:
        return self.path(key).with_suffix(".civ.json")

    def exists(self, key: RegionKey) -> bool:
        return self.path(key).is_file()

    def save(self, key: RegionKey, world, *, exclude_entity_id: int | None = None, society=None, adventure=None) -> dict:
        """Save a region without duplicating the travelling player inside it."""
        removed = None
        colony_memberships: list[int] = []
        feeding_state = None
        if exclude_entity_id is not None:
            removed = world.organisms.pop(int(exclude_entity_id), None)
            if removed is None:
                raise ValueError("region store excluded entity is missing")
            for colony_id, colony in world.colonies.items():
                if exclude_entity_id in colony.member_ids:
                    colony.member_ids.remove(exclude_entity_id)
                    colony_memberships.append(colony_id)
            if world.feeding_system is not None and hasattr(world.feeding_system,"entities"):
                feeding_state=world.feeding_system.entities.pop(int(exclude_entity_id),None)
        try:
            report = save_world(world, self.path(key))
        finally:
            if removed is not None:
                world.organisms[removed.entity_id] = removed
                for colony_id in colony_memberships:
                    world.colonies[colony_id].member_ids.add(removed.entity_id)
                if feeding_state is not None:world.feeding_system.entities[removed.entity_id]=feeding_state
        if society is not None:
            society_payload=_society_payload(society)
            if exclude_entity_id is not None:society_payload["assignments"].pop(str(exclude_entity_id),None);society_payload["assignments"].pop(exclude_entity_id,None)
            payload={"format":"nullvector-region-civilization/1.0.0","atlas_seed":self.atlas_seed,"region":[key.x,key.y,key.depth],"world_sha256":report["sha256"],"society":society_payload,"adventure_region":None if adventure is None else _adventure_payload(adventure)};payload["semantic_sha256"]=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest();path=self.civilization_path(key);path.parent.mkdir(parents=True,exist_ok=True);temporary=path.parent/("."+path.name+".tmp-"+uuid.uuid4().hex);temporary.write_text(json.dumps(payload,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n",encoding="utf-8");os.replace(temporary,path)
        return {**report, "region": (key.x, key.y, key.depth), "excluded": exclude_entity_id,"civilization":society is not None,"adventure":adventure is not None}

    def load(self, key: RegionKey, *, motion_policy=None, behavior_policy=None, colony_policy=None, feeding_system=None, society_policy=None, include_society:bool=False):
        path = self.path(key)
        if not path.is_file():
            return None
        world=load_world(path, motion_policy=motion_policy, behavior_policy=behavior_policy, colony_policy=colony_policy, feeding_system=feeding_system)
        if not include_society:return world
        civilization=self.civilization_path(key)
        if not civilization.is_file():return world,None,None
        payload=json.loads(civilization.read_text("utf-8"));claimed=payload.pop("semantic_sha256",None);actual=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
        if payload.get("format")!="nullvector-region-civilization/1.0.0" or claimed!=actual or payload.get("atlas_seed")!=self.atlas_seed or payload.get("region")!=[key.x,key.y,key.depth] or payload.get("world_sha256")!=hashlib.sha256(path.read_bytes()).hexdigest():raise ValueError("region civilization provenance drifted")
        regional_adventure=None if payload.get("adventure_region") is None else _restore_adventure(payload["adventure_region"]);return world,_restore_society(payload["society"],world,society_policy),regional_adventure
