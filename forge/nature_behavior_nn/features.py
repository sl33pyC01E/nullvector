from __future__ import annotations

import math

import numpy as np

from ..creature_stage_developmental import APPENDAGE_KINDS,COMPONENT_KINDS,FAMILIES
from ..nature_sim_v2.contract import INTENTS,LIFE_STAGES,RESOURCE_NAMES
from ..nature_sim_v2.state import OrganismState
from .contract import MAX_NEIGHBORS,NEIGHBOR_FEATURES,RESOURCE_FEATURES,SELF_FEATURES


ACTIVE_STAGES=LIFE_STAGES[:4]


def _one_hot(index:int,count:int)->np.ndarray:
    value=np.zeros(count,dtype=np.float32);value[index]=1;return value


def extract_observation(world,entity:OrganismState,system_cache:dict[int,dict[str,float]]|None=None)->tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    systems=entity.body.systems() if system_cache is None else system_cache[entity.entity_id];stage=ACTIVE_STAGES.index(entity.stage) if entity.stage in ACTIVE_STAGES else 0
    colony=world.colonies.get(entity.colony_id);colony_delta=np.zeros(2,dtype=np.float32) if colony is None else (world._delta(entity.position,colony.center)/world.size).astype(np.float32)
    appendage_counts=np.zeros(len(APPENDAGE_KINDS),np.float32)
    for item in entity.genome.developmental.appendages:appendage_counts[APPENDAGE_KINDS.index(item.kind)]+=1/8
    component_counts=np.zeros(len(COMPONENT_KINDS),np.float32)
    for item in entity.genome.developmental.components:component_counts[COMPONENT_KINDS.index(item.kind)]+=1/8
    self_features=np.concatenate((
        _one_hot(entity.family,len(FAMILIES)),np.asarray(entity.genome.developmental.traits,np.float32),
        np.asarray(entity.genome.eco_traits,np.float32),np.asarray(entity.genome.diet,np.float32),_one_hot(stage,len(ACTIVE_STAGES)),
        np.asarray([systems[name] for name in ("integrity","neural","circulation","respiration","digestion","senses","locomotion")],np.float32),
        np.asarray((entity.energy,entity.reserve,min(1,entity.age/400),min(1,entity.reproduction_cooldown/42),min(1,entity.gestation_remaining/20)),np.float32),
        np.asarray(entity.velocity,np.float32)/3.2,np.asarray((float(colony is not None),*colony_delta),np.float32),appendage_counts,component_counts,
        np.asarray((math.sin(entity.genome.developmental.seed*.0001),math.cos(entity.genome.developmental.seed*.0001),math.sin(world.time*.17),math.cos(world.time*.17),math.sin(entity.entity_id*.71),math.cos(entity.entity_id*.71)),np.float32),
    )).astype(np.float32)
    if self_features.shape!=(SELF_FEATURES,):raise AssertionError(f"behavior self feature drift: {self_features.shape}")
    y,x=world._cell(entity.position);resource=np.zeros((len(RESOURCE_NAMES),RESOURCE_FEATURES),np.float32)
    for index in range(len(RESOURCE_NAMES)):
        gradient=world._local_gradient(index,entity.position)
        resource[index]=(world.fields[index,y,x],gradient[0]*4,gradient[1]*4,entity.genome.diet[index])
    neighbors=sorted(world._neighbors(entity,world.size*.5),key=lambda other:(np.linalg.norm(world._delta(entity.position,other.position)),other.entity_id))[:MAX_NEIGHBORS]
    neighbor=np.zeros((MAX_NEIGHBORS,NEIGHBOR_FEATURES),np.float32);mask=np.zeros(MAX_NEIGHBORS,np.bool_)
    for index,other in enumerate(neighbors):
        delta=world._delta(entity.position,other.position);distance=float(np.linalg.norm(delta));direction=delta/max(distance,1e-6)
        other_systems=other.body.systems() if system_cache is None else system_cache[other.entity_id]
        values=np.concatenate((_one_hot(other.family,len(FAMILIES)),direction.astype(np.float32),np.asarray((min(1,distance/(world.size*.5)),other.energy,other.reserve,other_systems["integrity"],float(other.family!=entity.family),float(other.family==entity.family),float(other.stage=="mature")),np.float32)))
        neighbor[index]=values;mask[index]=True
    return self_features,resource,neighbor,mask


def intent_index(name:str)->int:return INTENTS.index(name)
