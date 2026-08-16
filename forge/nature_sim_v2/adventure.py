from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import numpy as np

from ..powder_world_v1.contract import MATERIALS,STATE
from ..qud_society_v1.architecture import generate_building
from ..qud_items_v1 import Artifact,RECIPES,craft,generate_artifact
from ..qud_encounters_v1 import SiteEncounter,generate_encounter,resolve_encounter


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
        self.artifacts:list[Artifact]=[];self.equipped:dict[str,str]={};self.recipe_index=0;self.craft_count=0
        self.encounters:dict[str,SiteEncounter]={};self.pending_encounter:str|None=None;self.encounters_completed=0;self.succession_count=0
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
        corpses=[other for other in world.organisms.values() if other.entity_id!=entity.entity_id and not other.alive and other.body.alive_mask.any()]
        if corpses:
            corpse=min(corpses,key=lambda other:float(np.linalg.norm(world._delta(entity.position,other.position))));distance=float(np.linalg.norm(world._delta(entity.position,corpse.position)))
            if distance<=2.6:
                alive_before=corpse.body.alive_mask.copy();owners=corpse.body.component_owner[alive_before];owner=int(np.bincount(owners).argmax());component=corpse.genome.developmental.components[owner];cx,cy=component.anchor;rx=max(1.0,float(component.radius[0]));corpse.body.cut((cx-rx,cy),(cx+rx,cy),width=1.15);removed=int(np.count_nonzero(alive_before&~corpse.body.alive_mask))
                if removed<=0:corpse.body.impact(component.anchor,max(1.2,float(max(component.radius))*.55),1.0);removed=int(np.count_nonzero(alive_before&~corpse.body.alive_mask))
                material=("biomass","biomass","biomass","crystal","metal")[corpse.family];amount=max(.01,removed*.012);self.inventory[material]+=amount
                if corpse.family==2:self.inventory["water"]+=amount*.35
                if component.organ in ("brain","phase_brain","processor","meristem"):self.inventory["knowledge"]+=amount*.18
                self.score+=max(1,removed//4);world.events.append({"tick":world.tick_index,"type":"corpse_harvest","harvester":entity.entity_id,"corpse":corpse.entity_id,"cells":removed,"material":material,"organ":component.organ});return f"BUTCHERED {removed} CELLS // +{amount:.2f} {material.upper()} // {component.organ.upper()}"
        nearest=min(self.sites,key=lambda site:np.linalg.norm(world._delta(entity.position,site.position)))
        distance=float(np.linalg.norm(world._delta(entity.position,nearest.position)))
        if distance<=2.6 and nearest.richness>.02:
            amount=min(.35,nearest.richness);nearest.richness-=amount
            rewards={"grove":("biomass",1.8),"mineral_vent":("rock",2.2),"machine_ruin":("metal",1.7),"phase_well":("crystal",1.25),"spring":("water",2.4),"relic_vault":("knowledge",1.4)}
            material,gain=rewards[nearest.kind];self.inventory[material]+=amount*gain
            first=not nearest.discovered
            if first:nearest.discovered=True;self.discoveries.add(nearest.site_id);self._advance("discover",1);self.score+=20
            if first:
                encounter=generate_encounter(seed=self.seed,site_id=nearest.site_id,kind=nearest.kind);self.encounters[encounter.encounter_id]=encounter;self.pending_encounter=encounter.encounter_id
            if first and nearest.kind in ("machine_ruin","phase_well","relic_vault"):
                site_seed=int(hashlib.sha256(f"{self.seed}:{nearest.site_id}:relic".encode()).hexdigest()[:16],16)
                artifact=generate_artifact(seed=site_seed,provenance=nearest.site_id,quality=min(1,.42+nearest.richness*.3));self.artifacts.append(artifact);self.equip(artifact.artifact_id)
                return f"SALVAGED {amount*gain:.2f} {material.upper()} // ENCOUNTER {encounter.title.upper()} // 4/5/6 CHOOSE // RELIC {artifact.name.upper()}"
            if first:return f"SALVAGED {amount*gain:.2f} {material.upper()} // ENCOUNTER {encounter.title.upper()} // 4/5/6 CHOOSE"
            if self.pending_encounter:
                pending=self.encounters[self.pending_encounter]
                return f"ENCOUNTER PENDING // {pending.title.upper()} // 4/5/6 CHOOSE"
            return f"HARVESTED {amount*gain:.2f} {material.upper()}"
        y,x=world._cell(entity.position);index=int(world.materials.material[y,x])
        if index and STATE[index]!="energy" and world.materials.mass[y,x]>.02:
            amount=min(.25,float(world.materials.mass[y,x]));name=MATERIALS[index];world.materials.mass[y,x]-=amount
            inventory_name=name if name in self.inventory else "biomass";self.inventory[inventory_name]+=amount
            return f"COLLECTED {amount:.2f} {name.upper()}"
        amount=min(.14,float(world.fields[8,y,x]+world.fields[9,y,x]));world.fields[8,y,x]=max(0,world.fields[8,y,x]-amount*.6);world.fields[9,y,x]=max(0,world.fields[9,y,x]-amount*.4);self.inventory["biomass"]+=amount
        return f"FORAGED {amount:.2f} BIOMASS"

    def resolve_pending(self,index:int,world,entity)->str:
        if self.pending_encounter is None:raise ValueError("no pending encounter")
        encounter=self.encounters[self.pending_encounter];result=resolve_encounter(encounter,index,world=world,entity=entity,adventure=self);self.pending_encounter=None;self.encounters_completed+=1;return result

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

    @property
    def selected_recipe(self):
        return RECIPES[self.recipe_index%len(RECIPES)]

    def cycle_recipe(self,delta:int=1)->str:
        self.recipe_index=(self.recipe_index+delta)%len(RECIPES);recipe=self.selected_recipe
        return f"RECIPE // {recipe.name.upper()} // "+" + ".join(f"{value:g} {name.upper()}" for name,value in recipe.costs)

    def craft_selected(self)->str:
        recipe=self.selected_recipe;seed=int(hashlib.sha256(f"{self.seed}:craft:{self.craft_count}:{recipe.recipe_id}".encode()).hexdigest()[:16],16)
        artifact=craft(recipe,seed=seed,provenance=f"field-craft-{self.craft_count}",inventory=self.inventory);self.craft_count+=1;self.artifacts.append(artifact);self.equip(artifact.artifact_id);self.score+=15
        return f"CRAFTED // {artifact.name.upper()} // {artifact.slot.upper()}"

    def equip(self,artifact_id:str)->str:
        artifact=next((item for item in self.artifacts if item.artifact_id==artifact_id),None)
        if artifact is None:raise ValueError("unknown artifact")
        self.equipped[artifact.slot]=artifact.artifact_id
        return artifact.name

    def equipped_artifacts(self)->tuple[Artifact,...]:
        lookup={item.artifact_id:item for item in self.artifacts}
        return tuple(lookup[item_id] for _,item_id in sorted(self.equipped.items()) if item_id in lookup)

    def bonus(self,name:str)->float:
        return sum(item.effect(name)*(.35+.65*item.durability) for item in self.equipped_artifacts())

    def abrade(self,slot:str,amount:float)->None:
        item_id=self.equipped.get(slot)
        for index,item in enumerate(self.artifacts):
            if item.artifact_id==item_id:
                self.artifacts[index]=Artifact(item.artifact_id,item.name,item.slot,item.material,item.components,item.effects,item.quality,max(0,item.durability-amount),item.seed,item.provenance);break
