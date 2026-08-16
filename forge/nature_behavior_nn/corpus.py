from __future__ import annotations

from dataclasses import dataclass
import hashlib,json,os
from pathlib import Path

import numpy as np

from ..nature_sim_v2 import INTENTS,NatureWorld
from .contract import FORMAT,MAX_NEIGHBORS,NEIGHBOR_FEATURES,RESOURCE_FEATURES,SELF_FEATURES,corpus_source_sha256
from .features import extract_observation,intent_index


@dataclass(frozen=True,slots=True)
class BehaviorCorpus:
    self_features:np.ndarray
    resource:np.ndarray
    neighbor:np.ndarray
    neighbor_mask:np.ndarray
    intent:np.ndarray
    direction:np.ndarray
    world_id:np.ndarray
    semantic_sha256:str


def _hash_arrays(arrays:dict[str,np.ndarray])->str:
    digest=hashlib.sha256(b"nullvector-nature-behavior-arrays-v1\0")
    for name in sorted(arrays):
        value=np.ascontiguousarray(arrays[name]);digest.update(name.encode()+b"\0"+str(value.dtype).encode()+b"\0"+np.asarray(value.shape,dtype="<i8").tobytes()+value.tobytes())
    return digest.hexdigest()


def build_corpus(output:Path,*,worlds:int=12,steps:int=260,seed:int=0x4245484156494F52)->dict[str,object]:
    if not 4<=worlds<=64 or not 80<=steps<=2000:raise ValueError("behavior corpus bounds drifted")
    buckets={name:[] for name in ("self_features","resource","neighbor","neighbor_mask","intent","direction","world_id")}
    for world_index in range(worlds):
        world=NatureWorld(seed=seed+world_index*104729,size=48,max_population=96);world.seed_founders(variants_per_family=3)
        for step in range(steps):
            # Deterministic scenario curriculum.  These remain genuine calls to
            # the complete scaffold policy, but avoid spending corpus-build
            # time on metabolism/body diffusion that does not affect labels.
            mode=step%6
            for entity in world.organisms.values():
                if mode in (1,3):entity.position=(np.asarray((24,24))+world.rng.normal(0,2.2,2))%world.size
                elif mode==2:
                    angle=entity.family*np.pi*2/5;entity.position=(np.asarray((24+np.cos(angle)*10,24+np.sin(angle)*10))+world.rng.normal(0,1.5,2))%world.size
                else:entity.position=(entity.position+world.rng.normal(0,3.5,2))%world.size
                energy=(.08,.28,.52,.77,1.0)[(step+entity.entity_id*3)%5];entity.energy=float(energy);entity.reserve=float(.12+.68*((step*7+entity.entity_id)%17)/16);entity.age=float(8+((step*13+entity.entity_id*19)%280));entity.reproduction_cooldown=0 if step%4 else 12;entity.update_stage()
            if step%24==0:world._update_colonies()
            system_cache={entity.entity_id:entity.body.systems() for entity in world.organisms.values() if entity.alive}
            for entity in sorted(world.organisms.values(),key=lambda item:item.entity_id):
                if not entity.alive:continue
                self_f,resource,neighbor,mask=extract_observation(world,entity,system_cache);raw=world._choose_intent(entity);norm=float(np.linalg.norm(raw));direction=np.zeros(2,np.float32) if norm<1e-8 else (raw/norm).astype(np.float32)
                for name,value in (("self_features",self_f),("resource",resource),("neighbor",neighbor),("neighbor_mask",mask),("intent",intent_index(entity.intent)),("direction",direction),("world_id",world_index)):buckets[name].append(value)
            world.tick_index+=1;world.time+=.25
    arrays={name:np.asarray(values,dtype={"neighbor_mask":np.bool_,"intent":np.uint8,"world_id":np.uint16}.get(name,np.float32)) for name,values in buckets.items()}
    digest=_hash_arrays(arrays);output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);stage=output.with_suffix(output.suffix+f".tmp-{os.getpid()}")
    np.savez_compressed(stage,**arrays,format=np.asarray(FORMAT),source_sha256=np.asarray(corpus_source_sha256()),semantic_sha256=np.asarray(digest));actual=stage if stage.exists() else Path(str(stage)+".npz");os.replace(actual,output)
    counts=np.bincount(arrays["intent"],minlength=len(INTENTS));report={"format":FORMAT,"source_sha256":corpus_source_sha256(),"semantic_sha256":digest,"samples":len(arrays["intent"]),"worlds":worlds,"steps":steps,"intent_counts":{name:int(counts[index]) for index,name in enumerate(INTENTS)},"artifact":output.name,"artifact_sha256":hashlib.sha256(output.read_bytes()).hexdigest()}
    output.with_suffix(".json").write_text(json.dumps(report,sort_keys=True,indent=2)+"\n","utf-8");return report


def load_corpus(path:Path)->BehaviorCorpus:
    with np.load(path,allow_pickle=False) as archive:
        if str(archive["format"])!=FORMAT or str(archive["source_sha256"])!=corpus_source_sha256():raise ValueError("behavior corpus provenance drifted")
        arrays={name:archive[name] for name in ("self_features","resource","neighbor","neighbor_mask","intent","direction","world_id")};digest=str(archive["semantic_sha256"])
    expected=(("self_features",(None,SELF_FEATURES),np.float32),("resource",(None,10,RESOURCE_FEATURES),np.float32),("neighbor",(None,MAX_NEIGHBORS,NEIGHBOR_FEATURES),np.float32),("neighbor_mask",(None,MAX_NEIGHBORS),np.bool_),("intent",(None,),np.uint8),("direction",(None,2),np.float32),("world_id",(None,),np.uint16))
    count=len(arrays["intent"])
    for name,shape,dtype in expected:
        value=arrays[name]
        if value.dtype!=dtype or value.shape!=(count,*shape[1:]) or not np.isfinite(value).all():raise ValueError(f"behavior corpus {name} drifted")
    if _hash_arrays(arrays)!=digest:raise ValueError("behavior corpus semantic hash drifted")
    return BehaviorCorpus(**arrays,semantic_sha256=digest)
