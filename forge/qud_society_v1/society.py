from __future__ import annotations

import hashlib
import json
import math

import numpy as np

from ..nature_sim_v2 import NatureWorld
from .architecture import expand_settlement,generate_building
from .contract import ACTIVITIES,CULTURAL_TRAITS,TECHNOLOGIES,Activity,FactionState,HistoryEvent,SettlementState


PREFIX=("Null","Verdant","Glass","Iron","Many","Quiet","Saffron","Spiral","Deep","Bright")
SUFFIX=("Concord","Kin","Assembly","Choir","Enclave","Caravan","Collective","Ward","Archive","Brood")


class SocietyLayer:
    def __init__(self,nature:NatureWorld,*,seed:int=0x515544)->None:
        self.nature=nature;self.seed=int(seed);self.rng=np.random.default_rng(seed);self.factions:dict[str,FactionState]={};self.settlements:dict[str,SettlementState]={};self.history:list[HistoryEvent]=[];self.activities:dict[str,Activity]={};self.assignments:dict[int,str]={};self.materialized_buildings:set[str]=set();self.tick=0

    @staticmethod
    def _initial_stockpiles(members)->dict[str,float]:
        return {"water":sum(float(item.consumed[0]) for item in members)+1.0,"flora":sum(float(item.consumed[8]) for item in members)+.5,"biomass":sum(float(item.consumed[9]) for item in members)+.5,"mineral":sum(float(item.consumed[2]) for item in members)+.4,"metal":.2,"crystal":sum(float(item.consumed[4]) for item in members)*.2,"food":sum(item.reserve for item in members)*.3,"medicine":0.0,"parts":0.0,"energy":sum(float(item.consumed[3]) for item in members)*.2,"knowledge":.1}

    def _harvest_environment(self,settlement:SettlementState,members:list)->None:
        """Move finite local field resources into a settlement's physical economy."""
        cx,cy=map(int,settlement.center);radius=max(3,min(8,2+len(members)//2));ys=(np.arange(cy-radius,cy+radius+1)%self.nature.size).astype(int);xs=(np.arange(cx-radius,cx+radius+1)%self.nature.size).astype(int);worker=max(1,len(members));rates={"water":(0,.010),"mineral":(2,.007),"energy":(3,.005),"crystal":(4,.003),"flora":(8,.010),"biomass":(9,.006)}
        for name,(channel,rate) in rates.items():
            patch=self.nature.fields[channel][np.ix_(ys,xs)];available=float(patch.sum());amount=min(available,rate*worker*(1+.08*len(settlement.buildings)));settlement.stockpiles[name]=settlement.stockpiles.get(name,0.0)+amount
            if available>1e-9:self.nature.fields[channel][np.ix_(ys,xs)]=patch*max(0,1-amount/available)
        settlement.production={name:0.0 for name in ("food","medicine","parts","energy","knowledge")}

    def _process_buildings(self,faction:FactionState,settlement:SettlementState)->None:
        stock=settlement.stockpiles
        def convert(output:str,amount:float,**inputs:float)->float:
            scale=min((stock.get(name,0.0)/need for name,need in inputs.items()),default=1.0);made=max(0,min(amount,amount*scale))
            if made<=0:return 0.0
            ratio=made/amount
            for name,need in inputs.items():stock[name]=max(0,stock.get(name,0.0)-need*ratio)
            stock[output]=stock.get(output,0.0)+made;settlement.production[output]=settlement.production.get(output,0.0)+made;return made
        for building in settlement.buildings:
            if building.purpose=="granary":convert("food",.18,flora=.12,water=.05)
            elif building.purpose=="clinic":convert("medicine",.07,food=.04,water=.05,biomass=.03)
            elif building.purpose in ("workshop","graft_house"):convert("parts",.08,mineral=.07,biomass=.02)
            elif building.purpose=="battery_hall":convert("energy",.12,mineral=.03)
            elif building.purpose=="observatory":convert("knowledge",.04,energy=.025)
            elif building.purpose=="market":settlement.wealth+=.012+stock.get("parts",0)*.001
            elif building.purpose=="habitat":convert("food",.04,flora=.03,water=.015)
        consumption=.012*max(1,settlement.population);eaten=min(stock.get("food",0),consumption);stock["food"]=max(0,stock.get("food",0)-eaten);settlement.food=stock.get("food",0)
        if eaten+1e-9<consumption:
            settlement.shortages+=1;faction.cohesion=max(.1,faction.cohesion-.025);settlement.wealth=max(0,settlement.wealth-.015)
        else:faction.cohesion=min(1,faction.cohesion+.004)
        knowledge=stock.get("knowledge",0);transfer=min(knowledge,.025);stock["knowledge"]-=transfer;faction.knowledge+=transfer;settlement.power=stock.get("energy",0)

    def _invest_surplus(self,faction:FactionState,settlement:SettlementState)->None:
        stock=settlement.stockpiles
        if len(settlement.buildings)>=min(24,max(4,settlement.population+3)) or stock.get("mineral",0)<.65 or stock.get("parts",0)<.18:return
        index=len(settlement.buildings);angle=index*2.399963229728653;radius=9+math.sqrt(index)*16;origin=(int(round(settlement.center[0]+math.cos(angle)*radius)),int(round(settlement.center[1]+math.sin(angle)*radius)));purpose=("granary","clinic","workshop","habitat","observatory","battery_hall","market","graft_house")[index%8];building=generate_building(seed=self.seed+self.tick*65537+index*7919,origin=origin,purpose=purpose)
        settlement.buildings.append(building);stock["mineral"]-=.65;stock["parts"]-=.18;settlement.projects_completed+=1;self._materialize(building);self.history.append(HistoryEvent(self.tick,"construction",(faction.faction_id,),settlement.center,f"{faction.name} completed a {purpose}.",(("buildings",1.0),)))

    def _materialize(self,building)->None:
        if building.building_id in self.materialized_buildings:return
        mask=np.zeros_like(self.nature.materials.material,dtype=np.bool_)
        for x,y,material in building.cells:
            if material=="wall":mask[y%self.nature.size,x%self.nature.size]=True
        structure_id=20_000+int(hashlib.sha256(building.building_id.encode()).hexdigest()[:7],16)%1_000_000
        self.nature.materials.add_structure(mask,structure_id=structure_id,material="rock");self.materialized_buildings.add(building.building_id)

    def _culture(self,colony_id:int,family:int)->tuple[float,...]:
        base=np.asarray((.64,.58,.34,.56,.30,.52,.72,.22),dtype=np.float64);base+=np.asarray(((0,.12,.08,.04,0,.08,.08,.02),(.05,-.02,.12,-.04,.12,-.04,.18,0),(.20,-.04,-.12,.18,.24,-.12,.30,.03),(-.18,.22,.08,-.14,.32,-.08,-.20,.48),(.30,.18,.10,-.04,-.12,.30,-.18,.05))[family]);base+=self.rng.normal(0,.06,len(CULTURAL_TRAITS));return tuple(np.clip(base,0,1))

    def found_from_colony(self,colony_id:int)->str:
        colony=self.nature.colonies[colony_id];members=[self.nature.organisms[i] for i in sorted(colony.member_ids) if i in self.nature.organisms]
        if len(members)<2:raise ValueError("society requires a persistent colony")
        digest=hashlib.sha256(f"{self.seed}:{colony_id}:{colony.founder_lineage}".encode()).hexdigest();faction_id=f"f-{digest[:12]}";name=f"{PREFIX[int(digest[:2],16)%len(PREFIX)]} {SUFFIX[int(digest[2:4],16)%len(SUFFIX)]}";culture=self._culture(colony_id,colony.family)
        technologies={"cell_cultivation"};
        if colony.family==0:technologies|={"organ_grafting","agriculture","medicine"}
        elif colony.family==4:technologies|={"machine_logic","architecture","mineral_tools"}
        elif colony.family==2:technologies|={"agriculture","medicine"}
        elif colony.family==3:technologies|={"phase_lensing"}
        center=tuple(map(float,colony.center));settlement_id=f"s-{digest[12:24]}";buildings,roads=expand_settlement(seed=int(digest[:12],16),center=center,count=max(3,min(12,len(members)+1)))
        settlement=SettlementState(settlement_id,faction_id,center,len(members),sum(o.reserve for o in members),sum(o.consumed[8]+o.consumed[9] for o in members),sum(o.consumed[3] for o in members),buildings,roads,self.nature.tick_index,self._initial_stockpiles(members))
        faction=FactionState(faction_id,name,colony.family,colony.founder_lineage,culture,technologies,{settlement_id},{},doctrine=("adapt","migrate","cultivate","transcend","construct")[colony.family]);self.factions[faction_id]=faction;self.settlements[settlement_id]=settlement
        for building in buildings:
            try:self._materialize(building)
            except ValueError:pass
        self.history.append(HistoryEvent(self.tick,"founding",(faction_id,),center,f"{name} founded {settlement_id}.",(("population",float(len(members))),)))
        return faction_id

    def discover_societies(self)->None:
        represented={f.lineage_id for f in self.factions.values()}
        for colony_id,colony in sorted(self.nature.colonies.items()):
            if colony.founder_lineage in represented or len(colony.member_ids)<3:continue
            if colony.family in (0,4) or len(colony.member_ids)>=6:self.found_from_colony(colony_id);represented.add(colony.founder_lineage)

    def _activity(self,faction:FactionState,settlement:SettlementState,index:int)->Activity:
        weights=np.asarray((.7,.3,.5,.2,.5,.8,.9,.7,.8,.7,.4,.5,.2,.2,.4,.2));weights[ACTIVITIES.index("build")]+=faction.cultural_traits[CULTURAL_TRAITS.index("industry")];weights[ACTIVITIES.index("study_anomaly")]+=faction.cultural_traits[-1];weights[ACTIVITIES.index("negotiate")]+=faction.cultural_traits[CULTURAL_TRAITS.index("mercy")]
        kind=ACTIVITIES[int(self.rng.choice(len(ACTIVITIES),p=weights/weights.sum()))];digest=hashlib.sha256(f"{self.seed}:{self.tick}:{faction.faction_id}:{index}:{kind}".encode()).hexdigest();difficulty=.15+.7*float(self.rng.random());reward=(("biomass",round(2+difficulty*8,2)),("knowledge",round(1+difficulty*5,2)))
        return Activity(f"a-{digest[:14]}",kind,faction.faction_id,settlement.center,difficulty,reward,f"{faction.name} requests: {kind.replace('_',' ')} near {settlement.settlement_id}.")

    def step_history(self,years:int=1)->None:
        for _ in range(years):
            self.tick+=1;self.discover_societies()
            for faction in sorted(self.factions.values(),key=lambda f:f.faction_id):
                faction.knowledge+=.015+.025*faction.cultural_traits[CULTURAL_TRAITS.index("curiosity")];faction.cohesion=float(np.clip(faction.cohesion+self.rng.normal(0,.01),.1,1))
                if faction.knowledge>.22+len(faction.technologies)*.08:
                    unknown=[t for t in TECHNOLOGIES if t not in faction.technologies]
                    if unknown:
                        technology=unknown[int(self.rng.integers(len(unknown)))];faction.technologies.add(technology);self.history.append(HistoryEvent(self.tick,"discovery",(faction.faction_id,),self.settlements[next(iter(faction.settlement_ids))].center,f"{faction.name} discovered {technology}.",(("knowledge",-0.05),)))
                for settlement_id in sorted(faction.settlement_ids):
                    settlement=self.settlements[settlement_id];settlement.wealth+=.02*len(faction.technologies);activity=self._activity(faction,settlement,0);self.activities[activity.activity_id]=activity
                    members=[entity for entity in self.nature.organisms.values() if entity.alive and (entity.genome.lineage_id==faction.lineage_id or (entity.colony_id in self.nature.colonies and self.nature.colonies[entity.colony_id].founder_lineage==faction.lineage_id))]
                    settlement.population=len(members);self._harvest_environment(settlement,members);self._process_buildings(faction,settlement);self._invest_surplus(faction,settlement)
                    for entity in members:self.assignments[entity.entity_id]=activity.activity_id
                    if "medicine" in faction.technologies and settlement.wealth>.05:
                        for entity in sorted(members,key=lambda item:item.body.systems()["integrity"])[:max(1,len(members)//4)]:entity.body.heal((0,0),12,.06)
                        settlement.wealth=max(0,settlement.wealth-.015*len(members))
                    for building in settlement.buildings:
                        try:self._materialize(building)
                        except ValueError:pass
            ids=sorted(self.factions)
            for i,left_id in enumerate(ids):
                for right_id in ids[i+1:]:
                    left,right=self.factions[left_id],self.factions[right_id];value=left.relations.get(right_id,0);compatibility=1-abs(left.cultural_traits[2]-right.cultural_traits[2]);value=float(np.clip(value+(compatibility-.5)*.02+self.rng.normal(0,.008),-1,1));left.relations[right_id]=value;right.relations[left_id]=value
            if len(self.activities)>256:self.activities=dict(list(sorted(self.activities.items()))[-256:])

    def semantic_sha256(self)->str:
        payload={"tick":self.tick,"factions":[(f.faction_id,f.name,f.family,f.cultural_traits,sorted(f.technologies),sorted(f.settlement_ids),sorted(f.relations.items()),f.knowledge,f.cohesion,f.doctrine) for f in sorted(self.factions.values(),key=lambda f:f.faction_id)],"settlements":[(s.settlement_id,s.faction_id,s.center,s.population,s.wealth,s.food,s.power,[b.building_id for b in s.buildings],sorted(s.roads),sorted(s.stockpiles.items()),sorted(s.production.items()),s.shortages,s.projects_completed) for s in sorted(self.settlements.values(),key=lambda s:s.settlement_id)],"history":[(h.tick,h.kind,h.actors,h.location,h.description,h.consequences) for h in self.history],"activities":[(a.activity_id,a.kind,a.issuer,a.location,a.difficulty,a.reward_materials,a.description) for a in sorted(self.activities.values(),key=lambda a:a.activity_id)],"assignments":sorted(self.assignments.items()),"materialized":sorted(self.materialized_buildings)}
        return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":"),default=list).encode()).hexdigest()
