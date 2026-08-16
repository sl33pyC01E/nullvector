from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True,slots=True)
class Ability:
    ability_id:str
    label:str
    energy_cost:float
    range:float
    kind:str


CATALOG={
    "bite":Ability("bite","cellular bite",.012,1.8,"melee"),
    "pounce":Ability("pounce","grounded pounce",.025,7,"movement"),
    "spore":Ability("spore","spore field",.04,5,"ecology"),
    "repair":Ability("repair","repair pulse",.035,4,"support"),
    "phase_fold":Ability("phase_fold","phase fold",.055,9,"anomaly"),
    "phase_wave":Ability("phase_wave","phase wave",.045,6,"anomaly"),
    "bolt":Ability("bolt","mineral bolt",.018,18,"ranged"),
    "beam":Ability("beam","cutting beam",.035,20,"ranged"),
}


def entity_abilities(entity,*,equipment_damage:float=0)->tuple[Ability,...]:
    appendages={item.kind for item in entity.genome.developmental.appendages};components={item.kind for item in entity.genome.developmental.components};organs={item.organ for item in entity.genome.developmental.components};result=[]
    defaults=(("pounce","repair"),("bite","pounce"),("spore","repair"),("phase_fold","phase_wave"),("bolt","beam"))[entity.family]
    for name in defaults:
        if name not in result:result.append(name)
    if appendages&{"leg","tail","tendril"} and "pounce" not in result:result.append("pounce")
    if appendages&{"root","frond"} and "spore" not in result:result.append("spore")
    if appendages&{"hardpoint","wheel"} or "generator" in components or equipment_damage>.05:
        if "bolt" not in result:result.append("bolt")
    if "weapon" in organs or "hardpoint" in appendages:
        if "beam" not in result:result.append("beam")
    if "phase" in organs or "orbital" in components:
        if "phase_fold" not in result:result.append("phase_fold")
    if organs&{"circulator","regenerator","heart"} and "repair" not in result:result.append("repair")
    return tuple(CATALOG[name] for name in result[:4])


def use_ability(world,entity,ability:Ability,target:tuple[float,float],*,power:float=1)->str:
    if not entity.alive or entity.body.incapacitated:return "ABILITY FAILED // BODY INCAPACITATED"
    if entity.energy<ability.energy_cost:return "ABILITY FAILED // INSUFFICIENT METABOLIC ENERGY"
    target_array=np.asarray(target,dtype=np.float64);delta=world._delta(entity.position,target_array);distance=float(np.linalg.norm(delta));direction=delta/max(distance,1e-8);entity.energy-=ability.energy_cost
    if ability.ability_id=="pounce":entity.velocity+=direction*(2.2+1.4*power);return "GROUNDED POUNCE // LIMB ANCHOR IMPULSE"
    if ability.ability_id=="bite":
        prey=[other for other in world._neighbors(entity,ability.range) if other.family!=entity.family]
        if not prey:return "BITE // NO BODY IN REACH"
        victim=min(prey,key=lambda other:np.linalg.norm(world._delta(entity.position,other.position)));normal=world._delta(victim.position,entity.position);point=victim.body.organism.cell_xy[int(np.argmin(victim.body.organism.cell_xy@(normal/max(float(np.linalg.norm(normal)),1e-8))))];cells=victim.body.impact(tuple(point),2.8,min(.28,.12*power));stolen=min(victim.energy,.035*power);victim.energy-=stolen;entity.energy=min(1.2,entity.energy+stolen*.75);return f"CELLULAR BITE // {cells} CELLS TRAUMATIZED"
    if ability.ability_id=="spore":
        y,x=world._cell(entity.position);radius=4;yy,xx=np.mgrid[:world.size,:world.size];dx=(xx-x+world.size//2)%world.size-world.size//2;dy=(yy-y+world.size//2)%world.size-world.size//2;weight=np.clip(1-np.hypot(dx,dy)/radius,0,1);world.fields[8]=np.clip(world.fields[8]+weight*.025*power,0,1);world.fields[5]=np.clip(world.fields[5]+weight*.006*power,0,1);world.materials.deposit("sap",tuple(entity.position),.16*power,2.5);return "SPORE FIELD // FLORA + OXYGEN + SAP"
    if ability.ability_id=="repair":
        healed=0
        for other in [entity]+world._neighbors(entity,ability.range):
            if other.family==entity.family:healed+=other.body.heal((0,0),12,.07*power)
        return f"REPAIR PULSE // {healed} CONNECTED CELLS"
    if ability.ability_id=="phase_fold":
        travel=min(distance,ability.range);destination=(entity.position+direction*travel)%world.size;y,x=world._cell(destination)
        if world.materials.structure_id[y,x]>0:return "PHASE FOLD BLOCKED // COHERENT STRUCTURE"
        origin=entity.position.copy();entity.position=destination;world.materials.deposit("crystal",tuple(origin),.08,1.2);world.materials.deposit("crystal",tuple(destination),.08,1.2);return f"PHASE FOLD // {travel:.1f} CELLS"
    if ability.ability_id=="phase_wave":
        affected=0
        for other in world._neighbors(entity,ability.range):
            if other.family!=entity.family:other.velocity+=world._delta(entity.position,other.position)/max(float(np.linalg.norm(world._delta(entity.position,other.position))),1e-8)*1.6*power;other.energy=max(0,other.energy-.025*power);affected+=1
        return f"PHASE WAVE // {affected} ORGANISMS DISPLACED"
    if ability.ability_id=="bolt":world.fire_projectile(entity.entity_id,tuple(target_array),speed=20+5*power,energy=2.2*power);return "MINERAL BOLT // PHYSICAL PROJECTILE"
    if ability.ability_id=="beam":
        result=world.fire_beam(entity.entity_id,tuple(target_array),energy=6*power,width=.65);return f"CUTTING BEAM // {result['cells_hit']} MATERIAL // {result['bodies_hit']} BODIES"
    raise ValueError("unknown anatomical ability")
