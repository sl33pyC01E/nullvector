from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ..living_body_substrate.contract import ORGAN_SYSTEM


SEASONS=("thaw","bright","dry","sporefall","long_dark","storm")


@dataclass(frozen=True,slots=True)
class ClimateState:
    season:str
    phase:float
    light:float
    rainfall:float
    heat:float
    phase_flux:float
    toxin:float
    event:str|None


class ClimateSystem:
    """Slow environmental forcing that creates heritable selection pressure."""
    def __init__(self,seed:int)->None:self.seed=int(seed);self.last_event_cycle=-1;self.current=self.sample(0)

    def sample(self,time:float)->ClimateState:
        period=180.0;cycle=time/period;phase=cycle%1;season_index=int(phase*len(SEASONS))%len(SEASONS);angle=phase*math.tau;raw=(self.seed^(int(cycle)*0x9E3779B1))&0xffffffff;event_index=(raw>>8)%5;event=(None,"drought","spore_bloom","mineral_upwelling","phase_storm")[event_index] if phase<.055 else None
        light=float(np.clip(.62+.34*math.sin(angle-.5),.15,1));rain=float(np.clip(.58+.38*math.sin(angle+1.35),.08,1));heat=float(np.clip(.48+.42*math.sin(angle-.9),.05,1));flux=float(np.clip(.14+.18*math.sin(angle*2+2.1),.01,.55));toxin=float(np.clip(.025+.035*math.sin(angle*3),0,.12))
        if event=="drought":rain*=.12;heat=min(1,heat+.28)
        elif event=="spore_bloom":rain=min(1,rain+.2);toxin=min(.4,toxin+.22)
        elif event=="phase_storm":flux=min(1,flux+.7)
        return ClimateState(SEASONS[season_index],phase,light,rain,heat,flux,toxin,event)

    @staticmethod
    def _organ_target(entity,system:str)->tuple[tuple[float,float],float]:
        candidates=[item for item in entity.genome.developmental.components if ORGAN_SYSTEM.get(item.organ)==system]
        component=candidates[0] if candidates else entity.genome.developmental.components[0]
        return component.anchor,max(1.0,float(max(component.radius))*.72)

    def stress_body(self,world,entity,state:ClimateState,delta:float)->tuple[str|None,float]:
        y,x=world._cell(entity.position);local_oxygen=float(world.fields[5,y,x]);local_heat=float(world.fields[6,y,x]);local_toxin=float(world.fields[7,y,x]);traits=entity.genome.developmental.traits
        exposures={
            "respiration":max(0,.24-local_oxygen)+max(0,local_toxin+state.toxin-.22)*(1-.5*traits[7]),
            "circulation":max(0,local_heat+state.heat*.25-.72)*(1-.45*traits[9]),
            "digestion":max(0,local_toxin+state.toxin-.34)*(1-.4*traits[10]),
            "neural":0 if entity.family==3 else max(0,state.phase_flux-.70)*(1-.55*traits[14]),
        }
        system,exposure=max(exposures.items(),key=lambda item:item[1])
        if exposure<=0:return None,0.0
        damage=min(.065,(.004+exposure*.032)*delta);anchor,radius=self._organ_target(entity,system);entity.body.impact(anchor,radius,damage);return system,damage

    def materialize_event(self,world,state:ClimateState,cycle:int)->int:
        if state.event not in ("spore_bloom","mineral_upwelling","phase_storm"):return 0
        rng=np.random.default_rng((self.seed^int(cycle)*0x9E3779B97F4A7C15)&0x7FFF_FFFF_FFFF_FFFF);count={"spore_bloom":14,"mineral_upwelling":9,"phase_storm":7}[state.event];materials={"spore_bloom":("biomass","sap"),"mineral_upwelling":("rock","crystal"),"phase_storm":("crystal",)}[state.event]
        for index in range(count):
            point=(float(rng.uniform(1,world.size-1)),float(rng.uniform(1,world.size-1)));material=materials[index%len(materials)];amount=float(rng.uniform(.025,.085));radius=float(rng.uniform(.55,1.8));world.materials.deposit(material,point,amount,radius)
        return count

    def step(self,world,delta:float)->ClimateState:
        state=self.sample(world.time);self.current=state;world.fields[1]=np.clip(world.fields[1]+(state.light-.55)*delta*.00018,0,1);world.fields[0]=np.clip(world.fields[0]+(state.rainfall-.5)*delta*.00016,0,1);world.fields[6]=np.clip(world.fields[6]+(state.heat-.45)*delta*.00012,0,1);world.fields[4]=np.clip(world.fields[4]+state.phase_flux*delta*.00007,0,1);world.fields[7]=np.clip(world.fields[7]+(state.toxin-.025)*delta*.00005,0,1)
        if state.event=="spore_bloom":world.fields[8]=np.clip(world.fields[8]+delta*.00035,0,1)
        elif state.event=="mineral_upwelling":world.fields[2]=np.clip(world.fields[2]+delta*.00028,0,1)
        for entity in world.organisms.values():
            if not entity.alive:continue
            traits=entity.genome.developmental.traits;drought=max(0,.28-state.rainfall)*(1-.55*traits[9]);toxin=state.toxin*(1-.45*traits[7]);phase=max(0,state.phase_flux-.5)*(0 if entity.family==3 else 1-.55*traits[14]);stress=(drought+toxin+phase)*delta*.0008;entity.energy=max(0,entity.energy-stress)
            if world.tick_index%15==entity.entity_id%15:
                system,damage=self.stress_body(world,entity,state,delta*15)
                if system is not None and damage>.006:world.events.append({"tick":world.tick_index,"type":"climate_injury","entity":entity.entity_id,"system":system,"damage":round(damage,6),"season":state.season,"event":state.event})
        cycle=int(world.time/180)
        if state.event and cycle!=self.last_event_cycle:
            deposits=self.materialize_event(world,state,cycle);world.events.append({"tick":world.tick_index,"type":"climate","event":state.event,"season":state.season,"cycle":cycle,"physical_deposits":deposits});self.last_event_cycle=cycle
        return state
