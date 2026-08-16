from __future__ import annotations

from dataclasses import dataclass
import hashlib,json
import numpy as np


BIOMES=("salt_steppe","fungal_garden","glass_dunes","flooded_archive","phase_reef","iron_wood","living_cavern","machine_grave")


@dataclass(frozen=True,slots=True)
class RegionKey:
    x:int
    y:int
    depth:int=0


@dataclass(frozen=True,slots=True)
class RegionSummary:
    key:RegionKey
    seed:int
    biome:str
    fertility:float
    mineral:float
    phase:float
    danger:float
    ruins:int
    population:tuple[int,...]
    visits:int=0
    world_sha256:str|None=None


class InfiniteNatureAtlas:
    """Lazy 64-bit region atlas with exact summaries for every visited chunk."""
    def __init__(self,*,seed:int)->None:self.seed=int(seed);self.visited:dict[RegionKey,RegionSummary]={}

    def _digest(self,key:RegionKey)->bytes:return hashlib.sha256(f"{self.seed}:{key.x}:{key.y}:{key.depth}".encode()).digest()

    def region_seed(self,key:RegionKey)->int:return int.from_bytes(self._digest(key)[:8],"little")&0x7FFF_FFFF_FFFF_FFFF

    def describe(self,key:RegionKey)->RegionSummary:
        if key in self.visited:return self.visited[key]
        raw=self._digest(key);unit=lambda offset:int.from_bytes(raw[offset:offset+2],"little")/65535;biome=BIOMES[raw[8]%len(BIOMES)];population=tuple(1+raw[12+index]%7 for index in range(5))
        return RegionSummary(key,self.region_seed(key),biome,unit(14),unit(16),unit(18),unit(20),1+raw[22]%9,population)

    def record(self,key:RegionKey,world)->RegionSummary:
        generated=self.describe(key);snap=world.snapshot();summary=RegionSummary(key,generated.seed,generated.biome,generated.fertility,generated.mineral,generated.phase,generated.danger,generated.ruins,snap.family_counts,generated.visits+1,snap.semantic_sha256);self.visited[key]=summary;return summary

    def window(self,center:RegionKey,radius:int=4)->tuple[RegionSummary,...]:
        if not 1<=radius<=32:raise ValueError("atlas window radius drifted")
        return tuple(self.describe(RegionKey(center.x+x,center.y+y,center.depth)) for y in range(-radius,radius+1) for x in range(-radius,radius+1))

    def semantic_sha256(self)->str:
        payload=[(key.x,key.y,key.depth,value.biome,value.population,value.visits,value.world_sha256) for key,value in sorted(self.visited.items(),key=lambda pair:(pair[0].depth,pair[0].y,pair[0].x))];return hashlib.sha256(json.dumps(payload,separators=(",",":"),default=list).encode()).hexdigest()

