from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import numpy as np

from ..powder_world_v1.contract import MATERIALS,STATE
from ..qud_society_v1.architecture import generate_building


@dataclass(slots=True)
class WorldSite:
    site_id:str
    kind:str
    position:np.ndarray
    richness:float
    discovered:bool=False


@dataclass(slots=True)
class ObjectiveState:
    objective_id:str
    description:str
    target:float
    progress:float=0
    complete:bool=False


class AdventureState:
    """Playable expedition state layered over the persistent nature world."""
    SITE_KINDS=("grove","mineral_vent","machine_ruin","phase_well","spring","relic_vault")
    def __init__(self,*,seed:int,size:int)->None:
        self.seed,self.size=int(seed),int(size);self.rng=np.random.default_rng(seed);self.inventory={name:0.0 for name in ("biomass","rock","metal","crystal","water","knowledge")};self.discoveries:set[str]=set();self.buildings=[];self.last_event=0;self.score=0
        self.sites=[]
        for index in range(24):
            kind=self.SITE_KINDS[index%len(self.SITE_KINDS)];position=self.rng.uniform(3,size-3,2);digest=hashlib.sha256(f"{seed}:{index}:{kind}".encode()).hexdigest();self.sites.append(WorldSite(f"site-{digest[:10]}",kind,position,float(self.rng.uniform(.7,1.4))))
        self.objectives=[ObjectiveState("discover","Discover three living-world sites",3),ObjectiveState("graft","Perform a physical organ or locomotor graft",1),ObjectiveState("build","Build a conglomerated field shelter",1),ObjectiveState("survive","Stay biologically active for 300 ecology ticks",300)]

    def _objective(self,name:str)->ObjectiveState:return next(item for item in self.objectives if item.objective_id==name)

    def _advance(self,name:str,value:float)->None:
        item=self._objective(name)
        if item.complete:return
        item.progress=min(item.target,item.progress+value)
        if item.progress>=item.target:item.complete=True;self.score+=100;self.inventory["knowledge"]+=2

    def observe(self,world)->None:
        events=world.events[self.last_event:]
        for event in events:
            if event.get("type")=="graft":self._advance("graft",1)
        self.last_event=len(world.events);survive=self._objective("survive");survive.progress=min(survive.target,float(world.tick_index));survive.complete=survive.progress>=survive.target

    def interact(self,world,entity)->str:
        nearest=min(self.sites,key=lambda site:np.linalg.norm(world._delta(entity.position,site.position)))
        distance=float(np.linalg.norm(world._delta(entity.position,nearest.position)))
        if distance<=2.6 and nearest.richness>.02:
            amount=min(.35,nearest.richness);nearest.richness-=amount
            rewards={"grove":("biomass",1.8),"mineral_vent":("rock",2.2),"machine_ruin":("metal",1.7),"phase_well":("crystal",1.25),"spring":("water",2.4),"relic_vault":("knowledge",1.4)}
            material,gain=rewards[nearest.kind];self.inventory[material]+=amount*gain
            if not nearest.discovered:nearest.discovered=True;self.discoveries.add(nearest.site_id);self._advance("discover",1);self.score+=20
            return f"SALVAGED {nearest.kind.upper()} // +{amount*gain:.2f} {material.upper()}"
        y,x=world._cell(entity.position);index=int(world.materials.material[y,x])
        if index and STATE[index]!="energy" and world.materials.mass[y,x]>.02:
            amount=min(.25,float(world.materials.mass[y,x]));name=MATERIALS[index];world.materials.mass[y,x]-=amount
            inventory_name=name if name in self.inventory else "biomass";self.inventory[inventory_name]+=amount
            return f"COLLECTED {amount:.2f} {name.upper()}"
        amount=min(.14,float(world.fields[8,y,x]+world.fields[9,y,x]));world.fields[8,y,x]=max(0,world.fields[8,y,x]-amount*.6);world.fields[9,y,x]=max(0,world.fields[9,y,x]-amount*.4);self.inventory["biomass"]+=amount
        return f"FORAGED {amount:.2f} BIOMASS"

    def build(self,world,entity)->str:
        cost={"rock":1.4,"metal":.7,"biomass":.4}
        missing=[name for name,value in cost.items() if self.inventory[name]<value]
        if missing:return "BUILD NEEDS // "+" + ".join(name.upper() for name in missing)
        origin=(int(entity.position[0])+2,int(entity.position[1])+2);plan=generate_building(seed=self.seed+len(self.buildings)*7919,origin=origin,purpose="habitat");mask=np.zeros_like(world.materials.material,dtype=np.bool_)
        for x,y,material in plan.cells:
            if material=="wall":mask[y%world.size,x%world.size]=True
        world.materials.add_structure(mask,structure_id=10_000+len(self.buildings)+1,material="rock")
        for name,value in cost.items():self.inventory[name]-=value
        self.buildings.append(plan);self._advance("build",1);self.score+=50;return f"BUILT {plan.purpose.upper()} // {sum(mask.flat)} PHYSICAL WALL CELLS"

