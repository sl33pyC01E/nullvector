from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True,slots=True)
class SensoryField:
    range:float
    arc_radians:float
    radial:bool
    rays:int
    integrity:float


def sensory_field(entity,*,equipment_bonus:float=0)->SensoryField:
    integrity=float(entity.body.systems()["senses"]);perception=entity.genome.trait("perception");developmental=entity.genome.developmental.traits[13];base=(perception+developmental)*.5;family_range=(10,12,7,13,16)[entity.family];arc=(math.radians(100),math.radians(132),math.tau,math.tau,math.radians(62))[entity.family];radial=entity.family in (2,3);sensor_organs=sum(component.organ in ("sensor","neural") or component.kind in ("head","sensor_crown") for component in entity.genome.developmental.components);value=(2+family_range*base)*(0.12+.88*integrity)*(1+equipment_bonus);return SensoryField(round(value,6),arc,radial,max(3,min(16,3+sensor_organs*2)),integrity)


def visible_targets(world,entity,field:SensoryField)->tuple[int,...]:
    facing=np.asarray((math.cos(entity.heading),math.sin(entity.heading)),dtype=np.float64);result=[]
    for other in world.organisms.values():
        if other.entity_id==entity.entity_id or not other.alive:continue
        delta=world._delta(entity.position,other.position);distance=float(np.linalg.norm(delta))
        if distance>field.range:continue
        if not field.radial and distance>1e-8:
            angle=math.acos(float(np.clip(np.dot(delta/distance,facing),-1,1)))
            if angle>field.arc_radians*.5:continue
        result.append(other.entity_id)
    return tuple(sorted(result))
