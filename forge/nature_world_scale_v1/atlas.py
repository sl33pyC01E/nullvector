from __future__ import annotations

from dataclasses import dataclass
import hashlib,json
import numpy as np


BIOMES=("salt_steppe","fungal_garden","glass_dunes","flooded_archive","phase_reef","iron_wood","living_cavern","machine_grave")
BIOME_RESOURCE_SCALE={
    "salt_steppe":(.42,1.22,1.28,.92,.50,.82,1.34,.72,.34,.48),
    "fungal_garden":(1.18,.62,.72,.52,.70,1.04,.48,1.18,1.72,1.46),
    "glass_dunes":(.30,1.45,1.20,1.18,.74,.68,1.72,.86,.22,.38),
    "flooded_archive":(1.75,.64,.82,.78,.62,.88,.40,.66,1.18,1.12),
    "phase_reef":(.72,.92,.88,1.16,2.10,.54,1.12,1.34,.82,.76),
    "iron_wood":(.88,.76,1.52,1.22,.52,1.08,.70,.62,1.28,.96),
    "living_cavern":(1.20,.35,.92,.48,.86,1.22,.54,.96,1.48,1.36),
    "machine_grave":(.46,.82,1.82,1.94,.58,.48,1.26,1.18,.26,.42),
}


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

    def terraform(self,world,key:RegionKey)->RegionSummary:
        """Apply a region's ecology and organized ruins to a fresh world."""
        summary=self.describe(key);world.biome=summary.biome;scale=np.asarray(BIOME_RESOURCE_SCALE[summary.biome],dtype=np.float64)[:,None,None];world.fields[:]=np.clip(world.fields*scale,0,1);world.fields[2]*=.72+.68*summary.mineral;world.fields[4]*=.72+.92*summary.phase;world.fields[8]*=.62+.84*summary.fertility
        rng=np.random.default_rng(summary.seed^0x5255494E)
        for index in range(min(5,summary.ruins)):
            width=int(rng.integers(6,11));height=int(rng.integers(6,11));x=int(rng.integers(2,max(3,world.size-width-2)));y=int(rng.integers(2,max(3,world.size-height-2)));mask=np.zeros((world.size,world.size),dtype=np.bool_);mask[y:y+2,x:x+width]=True;mask[y:y+height,x:x+2]=True;mask[y+height-2:y+height,x:x+width]=True;mask[y:y+height,x+width-2:x+width]=True;opening=y+height//2;mask[opening-1:opening+2,x+width-2:x+width]=False
            try:world.materials.add_structure(mask,structure_id=30_000+index,material="metal" if summary.biome in ("machine_grave","flooded_archive") else "rock")
            except ValueError:pass
        return summary

    def window(self,center:RegionKey,radius:int=4)->tuple[RegionSummary,...]:
        if not 1<=radius<=32:raise ValueError("atlas window radius drifted")
        return tuple(self.describe(RegionKey(center.x+x,center.y+y,center.depth)) for y in range(-radius,radius+1) for x in range(-radius,radius+1))

    def semantic_sha256(self)->str:
        payload=[(key.x,key.y,key.depth,value.biome,value.population,value.visits,value.world_sha256) for key,value in sorted(self.visited.items(),key=lambda pair:(pair[0].depth,pair[0].y,pair[0].x))];return hashlib.sha256(json.dumps(payload,separators=(",",":"),default=list).encode()).hexdigest()
