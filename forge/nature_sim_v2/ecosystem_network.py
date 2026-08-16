from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True,slots=True)
class EcosystemLink:
    kind:str
    left:int
    right:int
    strength:float


class EcosystemNetwork:
    """Materially grounded mutualism, dispersal, scavenging, and symbiosis."""
    def __init__(self)->None:
        self.pollinations=0;self.seed_dispersals=0;self.root_transfers=0;self.scavenges=0;self.phase_couplings=0;self.links:tuple[EcosystemLink,...]=();self.last_event_tick:dict[tuple[str,int,int],int]={}

    def _ready(self,kind:str,left:int,right:int,tick:int,interval:int)->bool:
        key=(kind,min(left,right),max(left,right));last=self.last_event_tick.get(key,-interval)
        if tick-last<interval:return False
        self.last_event_tick[key]=tick;return True

    def step(self,world,delta:float)->None:
        alive=[item for item in sorted(world.organisms.values(),key=lambda value:value.entity_id) if item.alive];links=[]
        for left_index,left in enumerate(alive):
            for right in alive[left_index+1:]:
                displacement=world._delta(left.position,right.position);distance=float(np.linalg.norm(displacement))
                if distance>3.2:continue
                pair={left.family,right.family}
                if 2 in pair and (0 in pair or 1 in pair):
                    plant=left if left.family==2 else right;carrier=right if plant is left else left;strength=max(0,1-distance/3.2);links.append(EcosystemLink("pollination",plant.entity_id,carrier.entity_id,strength))
                    if self._ready("pollination",plant.entity_id,carrier.entity_id,world.tick_index,17):
                        plant.reserve=min(1,plant.reserve+.012*strength);plant.reproduction_cooldown=max(0,plant.reproduction_cooldown-.35*strength);carrier.energy=min(1,carrier.energy+.006*strength);self.pollinations+=1;world.events.append({"tick":world.tick_index,"type":"pollination","plant":plant.entity_id,"carrier":carrier.entity_id,"strength":round(strength,6)})
                        if (world.tick_index+plant.entity_id*7+carrier.entity_id*13)%3==0:
                            direction=displacement/max(distance,1e-6);target=(carrier.position+direction*2.5)%world.size;y,x=world._cell(target);world.fields[8,y,x]=min(1,world.fields[8,y,x]+.08+.08*strength);self.seed_dispersals+=1
                if left.family==right.family==2 and left.colony_id is not None and left.colony_id==right.colony_id:
                    strength=max(0,1-distance/3.2);links.append(EcosystemLink("root_network",left.entity_id,right.entity_id,strength))
                    if self._ready("root",left.entity_id,right.entity_id,world.tick_index,11):
                        energy=(left.energy+right.energy)*.5;reserve=(left.reserve+right.reserve)*.5;left.energy+=(energy-left.energy)*.18*strength;right.energy+=(energy-right.energy)*.18*strength;left.reserve+=(reserve-left.reserve)*.12*strength;right.reserve+=(reserve-right.reserve)*.12*strength;self.root_transfers+=1
                if pair=={3,4}:
                    anomaly=left if left.family==3 else right;machine=right if anomaly is left else left;strength=max(0,1-distance/3.2);links.append(EcosystemLink("phase_charge",anomaly.entity_id,machine.entity_id,strength))
                    if self._ready("phase",anomaly.entity_id,machine.entity_id,world.tick_index,13):
                        y,x=world._cell(anomaly.position);phase=min(.02*strength,float(world.fields[4,y,x]));world.fields[4,y,x]-=phase;machine.energy=min(1,machine.energy+phase*.8);anomaly.reserve=min(1,anomaly.reserve+phase*.25);self.phase_couplings+=1
        corpses=[item for item in world.organisms.values() if not item.alive and item.decomposition<1]
        for scavenger in alive:
            if scavenger.family not in (0,1,4) or scavenger.energy>.7:continue
            candidates=[corpse for corpse in corpses if float(np.linalg.norm(world._delta(scavenger.position,corpse.position)))<2.2]
            if not candidates:continue
            corpse=min(candidates,key=lambda value:value.entity_id);strength=max(0,1-float(np.linalg.norm(world._delta(scavenger.position,corpse.position)))/2.2);links.append(EcosystemLink("scavenge",scavenger.entity_id,corpse.entity_id,strength))
            if self._ready("scavenge",scavenger.entity_id,corpse.entity_id,world.tick_index,9):
                gain=min(.025*strength,max(0,1-corpse.decomposition)*.02);scavenger.energy=min(1,scavenger.energy+gain);scavenger.reserve=min(1,scavenger.reserve+gain*.6);corpse.decomposition=min(1,corpse.decomposition+gain*.3);self.scavenges+=1;world.events.append({"tick":world.tick_index,"type":"scavenge","entity":scavenger.entity_id,"corpse":corpse.entity_id,"gain":round(gain,6)})
        self.links=tuple(links)

    def payload(self)->dict:
        return {"pollinations":self.pollinations,"seed_dispersals":self.seed_dispersals,"root_transfers":self.root_transfers,"scavenges":self.scavenges,"phase_couplings":self.phase_couplings,"last_event_tick":[(*key,value) for key,value in sorted(self.last_event_tick.items())]}

    def restore(self,payload:dict)->None:
        self.pollinations=int(payload.get("pollinations",0));self.seed_dispersals=int(payload.get("seed_dispersals",0));self.root_transfers=int(payload.get("root_transfers",0));self.scavenges=int(payload.get("scavenges",0));self.phase_couplings=int(payload.get("phase_couplings",0));self.last_event_tick={(str(kind),int(left),int(right)):int(tick) for kind,left,right,tick in payload.get("last_event_tick",[])}
