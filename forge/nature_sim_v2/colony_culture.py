from __future__ import annotations

from dataclasses import dataclass
import hashlib
import numpy as np


CULTURE_TRAITS=("foraging","defense","medicine","construction","dispersal","broodcare")


@dataclass(slots=True)
class ColonyCultureState:
    colony_id:int
    values:np.ndarray
    generation:int=0
    parent_colony:int|None=None
    observations:int=0
    adaptations:int=0


class ColonyCultureSystem:
    """Heritable collective strategies learned from physical colony outcomes."""
    def __init__(self)->None:self.states:dict[int,ColonyCultureState]={};self.last_predation=0

    @staticmethod
    def _founder_values(members)->np.ndarray:
        if not members:return np.full(len(CULTURE_TRAITS),.5,np.float64)
        eco=np.mean([item.genome.eco_traits for item in members],axis=0);dev=np.mean([item.genome.developmental.traits for item in members],axis=0)
        return np.clip((.25+.5*eco[6],.2+.45*eco[9]+.2*dev[5],.2+.45*eco[8]+.2*dev[11],.2+.45*dev[3]+.2*dev[6],.2+.6*eco[14],.2+.4*eco[12]+.25*eco[3]),0,1)

    def ensure(self,world,colony_id:int)->ColonyCultureState:
        if colony_id in self.states:return self.states[colony_id]
        colony=world.colonies[colony_id];members=[world.organisms[item] for item in sorted(colony.member_ids) if item in world.organisms and world.organisms[item].alive];parents=[state for cid,state in self.states.items() if cid!=colony_id and cid in world.colonies and world.colonies[cid].founder_lineage==colony.founder_lineage and world.colonies[cid].generation<colony.generation]
        if parents:
            parent=max(parents,key=lambda item:(world.colonies[item.colony_id].generation,item.observations));digest=hashlib.sha256(f"{world.seed}:{colony_id}:{parent.colony_id}:culture".encode()).digest();noise=(np.frombuffer(digest[:6],dtype=np.uint8).astype(np.float64)/255-.5)*.08;values=np.clip(parent.values+noise,0,1);state=ColonyCultureState(colony_id,values,parent.generation+1,parent.colony_id)
        else:state=ColonyCultureState(colony_id,self._founder_values(members),colony.generation)
        self.states[colony_id]=state;return state

    def fission_threshold(self,world,colony_id:int)->int:
        state=self.ensure(world,colony_id);return int(np.clip(round(18-state.values[4]*7),10,18))

    def step(self,world,delta:float)->None:
        danger=float(world.predation_events>self.last_predation);self.last_predation=world.predation_events
        for colony_id,colony in sorted(world.colonies.items()):
            state=self.ensure(world,colony_id);members=[world.organisms[item] for item in sorted(colony.member_ids) if item in world.organisms and world.organisms[item].alive]
            if not members:continue
            state.observations+=1;mean_energy=float(np.mean([item.energy for item in members]));mean_integrity=float(np.mean([item.body.systems()["integrity"] for item in members]));culture_target=np.asarray((1-mean_energy,danger,1-mean_integrity,min(1,len(members)/14),float(len(members)>10),float(sum(item.stage in ("embryo","juvenile") for item in members)/len(members))),np.float64);rate=min(.012,delta*.01);before=state.values.copy();state.values=np.clip(state.values+(culture_target-state.values)*rate,0,1);state.adaptations+=int(np.max(np.abs(state.values-before))>1e-5)
            roles=world.colony_ecology.states.get(colony_id);assignments={} if roles is None else roles.assignments
            for entity in members:
                role=assignments.get(entity.entity_id)
                if role=="gatherer":entity.energy=min(1.2,entity.energy+delta*.00045*state.values[0])
                elif role=="defender":entity.reserve=min(1,entity.reserve+delta*.00022*state.values[1])
                elif role=="breeder":entity.reproduction_cooldown=max(0,entity.reproduction_cooldown-delta*.018*state.values[5])
            if world.tick_index%60==colony_id%60 and state.values[2]>.35:
                patient=min(members,key=lambda item:item.body.systems()["integrity"]);ecology=world.colony_ecology.states.get(colony_id);available=0 if ecology is None else ecology.energy_store;amount=min(.025*state.values[2],available)
                if amount>0:patient.body.energy=min(1.2,patient.body.energy+amount);patient.body.heal((0,0),8,amount);ecology.energy_store-=amount

    def payload(self)->dict:
        return {"last_predation":self.last_predation,"states":{str(key):{"values":value.values.tolist(),"generation":value.generation,"parent":value.parent_colony,"observations":value.observations,"adaptations":value.adaptations} for key,value in sorted(self.states.items())}}

    def restore(self,payload:dict)->None:
        self.last_predation=int(payload.get("last_predation",0));self.states={int(key):ColonyCultureState(int(key),np.asarray(value["values"],np.float64),int(value["generation"]),value["parent"],int(value["observations"]),int(value["adaptations"])) for key,value in payload.get("states",{}).items()}
