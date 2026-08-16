from __future__ import annotations

from dataclasses import asdict
import hashlib,json,os
from pathlib import Path
import uuid,zipfile

import numpy as np

from ..creature_stage_developmental import AppendageGene,ComponentGene,DevelopmentalGenome
from ..living_body_substrate import LivingBody
from ..powder_world_v1.contract import Projectile
from .colony_ecology import ColonyEcologyState
from .contract import EcoGenome
from .state import ColonyState,OrganismState
from .world import NatureWorld


SAVE_FORMAT="nullvector-living-nature-save/1.0.0"
MAX_SAVE_BYTES=768*1024**2


def _plain(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {str(key):_plain(item) for key,item in value.items()}
    if isinstance(value,(tuple,list)):return [_plain(item) for item in value]
    return value


def _genome_payload(genome:EcoGenome)->dict:
    return {"developmental":asdict(genome.developmental),"eco_traits":genome.eco_traits,"diet":genome.diet,"lineage_id":genome.lineage_id,"mutation_log":genome.mutation_log}


def _genome(value:dict)->EcoGenome:
    raw=value["developmental"];developmental=DevelopmentalGenome(raw["genome_id"],int(raw["seed"]),tuple(raw["family_mix"]),tuple(raw["traits"]),tuple(ComponentGene(**{**item,"anchor":tuple(item["anchor"]),"radius":tuple(item["radius"]),"trait_delta":tuple(item["trait_delta"])}) for item in raw["components"]),tuple(AppendageGene(**{**item,"root_offset":tuple(item["root_offset"]),"endpoint":tuple(item["endpoint"]),"trait_delta":tuple(item["trait_delta"])}) for item in raw["appendages"]),int(raw["generation"]),tuple(raw["parent_ids"]));return EcoGenome(developmental,tuple(value["eco_traits"]),tuple(value["diet"]),value["lineage_id"],tuple(value["mutation_log"]))


def save_world(world:NatureWorld,path:Path)->dict:
    path=Path(path).resolve();path.parent.mkdir(parents=True,exist_ok=True);arrays={"fields":world.fields,"material":world.materials.material,"mass":world.materials.mass,"temperature":world.materials.temperature,"damage":world.materials.damage,"structure_id":world.materials.structure_id};organisms=[]
    for entity in sorted(world.organisms.values(),key=lambda item:item.entity_id):
        prefix=f"e{entity.entity_id}_";arrays.update({prefix+"health":entity.body.health,prefix+"scar":entity.body.scar,prefix+"fluid":entity.body.fluid,prefix+"separation":entity.body.separation_age,prefix+"consumed":entity.consumed,prefix+"contacts":entity.neural_contacts,prefix+"muscles":entity.neural_muscles})
        body={"seed":entity.body.seed,"energy":entity.body.energy,"tick":entity.body.tick_index,"incapacitated":entity.body.incapacitated,"dead":entity.body.dead,"main_seed":entity.body.main_seed_cell,"puddles":_plain(entity.body.external_puddle),"polyps":_plain(entity.body.polyps),"biomass":_plain(entity.body.biomass)}
        organisms.append({"id":entity.entity_id,"genome":_genome_payload(entity.genome),"position":entity.position.tolist(),"velocity":entity.velocity.tolist(),"age":entity.age,"energy":entity.energy,"reserve":entity.reserve,"stage":entity.stage,"intent":entity.intent,"target":None if entity.target is None else entity.target.tolist(),"reproduction_cooldown":entity.reproduction_cooldown,"gestation":entity.gestation_remaining,"mate":entity.mate_id,"colony":entity.colony_id,"alive":entity.alive,"decomposition":entity.decomposition,"heading":entity.heading,"birth_tick":entity.birth_tick,"parents":entity.parent_ids,"polyp_cursor":entity.polyp_cursor,"body":body})
    colony_ecology={str(key):{"assignments":state.assignments,"energy_store":state.energy_store,"material_store":state.material_store.tolist(),"cohesion":state.cohesion,"transfers":state.transfers,"repairs":state.repairs} for key,state in world.colony_ecology.states.items()}
    metadata={"format":SAVE_FORMAT,"seed":world.seed,"size":world.size,"max_population":world.max_population,"biome":world.biome,"tick":world.tick_index,"time":world.time,"next_entity":world.next_entity_id,"next_colony":world.next_colony_id,"births":world.births,"deaths":world.deaths,"predation":world.predation_events,"mutations":world.mutation_count,"events":_plain(world.events),"rng":_plain(world.rng.bit_generator.state),"organisms":organisms,"colonies":[{"id":colony.colony_id,"family":colony.family,"lineage":colony.founder_lineage,"members":sorted(colony.member_ids),"center":colony.center.tolist(),"generation":colony.generation,"fissions":colony.fissions} for colony in sorted(world.colonies.values(),key=lambda item:item.colony_id)],"colony_ecology":colony_ecology,"colony_culture":world.colony_culture.payload(),"ecosystem":world.ecosystem.payload(),"breeding":world.breeding.payload(),"materials":{"tick":world.materials.tick_index,"next_projectile":world.materials.next_projectile,"projectiles":[asdict(item) for item in world.materials.projectiles]}}
    metadata_bytes=(json.dumps(metadata,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode();arrays["metadata_json"]=np.frombuffer(metadata_bytes,dtype=np.uint8);temporary=path.parent/("."+path.name+".tmp-"+uuid.uuid4().hex)
    try:
        with temporary.open("wb") as handle:np.savez_compressed(handle,**arrays)
        if temporary.stat().st_size>MAX_SAVE_BYTES:raise ValueError("nature save exceeds size bound")
        os.replace(temporary,path)
    finally:
        if temporary.exists():temporary.unlink()
    return {"path":str(path),"bytes":path.stat().st_size,"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"world_sha256":world.snapshot().semantic_sha256,"organisms":len(organisms)}


def load_world(path:Path,*,motion_policy=None,behavior_policy=None,colony_policy=None)->NatureWorld:
    path=Path(path).resolve()
    if not path.is_file() or path.stat().st_size>MAX_SAVE_BYTES:raise ValueError("nature save missing or oversized")
    with zipfile.ZipFile(path) as archive:
        if len(archive.infolist())>256 or any(info.file_size>MAX_SAVE_BYTES for info in archive.infolist()):raise ValueError("nature save archive bounds failed")
    with np.load(path,allow_pickle=False) as archive:
        arrays={name:np.ascontiguousarray(archive[name]) for name in archive.files}
    metadata=json.loads(arrays.pop("metadata_json").tobytes().decode())
    if metadata.get("format")!=SAVE_FORMAT:raise ValueError("nature save format drifted")
    world=NatureWorld(seed=int(metadata["seed"]),size=int(metadata["size"]),max_population=int(metadata["max_population"]),motion_policy=motion_policy,behavior_policy=behavior_policy);world.biome=metadata.get("biome");world.organisms.clear();world.colonies.clear();world.fields[:]=arrays.pop("fields");world.materials.material[:]=arrays.pop("material");world.materials.mass[:]=arrays.pop("mass");world.materials.temperature[:]=arrays.pop("temperature");world.materials.damage[:]=arrays.pop("damage");world.materials.structure_id[:]=arrays.pop("structure_id")
    for record in metadata["organisms"]:
        entity_id=int(record["id"]);genome=_genome(record["genome"]);body=LivingBody(world.organisms.get(entity_id,OrganismState.spawn(entity_id,genome,(0,0))).body.organism,seed=int(record["body"]["seed"]));prefix=f"e{entity_id}_";body.health[:]=arrays.pop(prefix+"health");body.scar[:]=arrays.pop(prefix+"scar");body.fluid[:]=arrays.pop(prefix+"fluid");body.separation_age[:]=arrays.pop(prefix+"separation");body.energy=float(record["body"]["energy"]);body.tick_index=int(record["body"]["tick"]);body.incapacitated=bool(record["body"]["incapacitated"]);body.dead=bool(record["body"]["dead"]);body.main_seed_cell=int(record["body"]["main_seed"]);body.external_puddle=record["body"]["puddles"];body.polyps=record["body"]["polyps"];body.biomass=record["body"]["biomass"]
        entity=OrganismState(entity_id,genome,body,np.asarray(record["position"],np.float64),np.asarray(record["velocity"],np.float64),age=float(record["age"]),energy=float(record["energy"]),reserve=float(record["reserve"]),stage=record["stage"],intent=record["intent"],target=None if record["target"] is None else np.asarray(record["target"],np.float64),reproduction_cooldown=float(record["reproduction_cooldown"]),gestation_remaining=float(record["gestation"]),mate_id=record["mate"],colony_id=record["colony"],alive=bool(record["alive"]),decomposition=float(record["decomposition"]),heading=float(record["heading"]),birth_tick=int(record["birth_tick"]),parent_ids=tuple(record["parents"]),consumed=arrays.pop(prefix+"consumed"),neural_contacts=arrays.pop(prefix+"contacts"),neural_muscles=arrays.pop(prefix+"muscles"),polyp_cursor=int(record["polyp_cursor"]));world.organisms[entity_id]=entity
    for record in metadata["colonies"]:world.colonies[int(record["id"])]=ColonyState(int(record["id"]),int(record["family"]),record["lineage"],set(record["members"]),np.asarray(record["center"],np.float64),int(record["generation"]),int(record["fissions"]))
    world.tick_index=int(metadata["tick"]);world.time=float(metadata["time"]);world.next_entity_id=int(metadata["next_entity"]);world.next_colony_id=int(metadata["next_colony"]);world.births=int(metadata["births"]);world.deaths=int(metadata["deaths"]);world.predation_events=int(metadata["predation"]);world.mutation_count=int(metadata["mutations"]);world.events=metadata["events"];world.rng.bit_generator.state=metadata["rng"];world.materials.tick_index=int(metadata["materials"]["tick"]);world.materials.next_projectile=int(metadata["materials"]["next_projectile"]);world.materials.projectiles=[Projectile(**{**item,"position":tuple(item["position"]),"velocity":tuple(item["velocity"])}) for item in metadata["materials"]["projectiles"]]
    world.colony_ecology.states={int(key):ColonyEcologyState(int(key),{int(entity_id):role for entity_id,role in value["assignments"].items()},float(value["energy_store"]),np.asarray(value["material_store"],np.float64),float(value["cohesion"]),int(value["transfers"]),int(value["repairs"])) for key,value in metadata["colony_ecology"].items()};world.colony_ecology.role_policy=colony_policy;world.colony_culture.restore(metadata.get("colony_culture",{}));world.ecosystem.restore(metadata.get("ecosystem",{}));world.breeding.restore(metadata.get("breeding",{}))
    if arrays:raise ValueError("nature save contains unconsumed arrays")
    return world
