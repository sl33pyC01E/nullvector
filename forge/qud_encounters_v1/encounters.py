from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np

from ..nature_sim_v2.genetics import founder_genomes
from ..qud_items_v1 import generate_artifact


@dataclass(frozen=True,slots=True)
class EncounterChoice:
    choice_id:str
    label:str
    approach:str
    risk:float
    reward_scale:float


@dataclass(slots=True)
class SiteEncounter:
    encounter_id:str
    site_id:str
    kind:str
    title:str
    description:str
    choices:tuple[EncounterChoice,...]
    resolved:bool=False
    outcome:str=""


TITLES={"grove":("The Many-Root Communion","A tessellated root intelligence folds nutrients through the soil."),"mineral_vent":("The Singing Fault","Pressurized crystal and hot mineral slurry pulse beneath a thin crust."),"machine_ruin":("The Dormant Fabricator","A geometric ruin watches through cracked lenses and live hardpoints."),"phase_well":("The Impossible Reflection","Your outline returns before you move and asks to exchange organs."),"spring":("The Vascular Spring","Mineral water carries free cells that seek compatible circulation."),"relic_vault":("The Sealed Memory","A cellular lock surrounds a relic assembled by an extinct lineage.")}


def generate_encounter(*,seed:int,site_id:str,kind:str)->SiteEncounter:
    if kind not in TITLES:raise ValueError("encounter site kind drifted")
    title,description=TITLES[kind];digest=hashlib.sha256(f"{seed}:{site_id}:{kind}:encounter".encode()).hexdigest();base={"grove":.20,"mineral_vent":.42,"machine_ruin":.52,"phase_well":.58,"spring":.16,"relic_vault":.64}[kind];choices=(EncounterChoice("observe","Observe and map","perception",base*.45,.65),EncounterChoice("enter","Enter bodily","integrity",base,1.25),EncounterChoice("interface","Interface with it","neural",base*.78,1.0));return SiteEncounter("enc-"+digest[:14],site_id,kind,title,description,choices)


def _capacity(entity,approach:str,adventure)->float:
    systems=entity.body.systems();traits=entity.genome.developmental.traits
    if approach=="perception":return systems["senses"]*.55+traits[13]*.25+entity.genome.trait("perception")*.2+adventure.bonus("perception")*.2
    if approach=="neural":return systems["neural"]*.55+traits[8]*.25+traits[14]*.2+adventure.bonus("phase")*.2
    return systems["integrity"]*.55+traits[3]*.2+traits[5]*.15+adventure.bonus("armor")*.22


def resolve_encounter(encounter:SiteEncounter,choice_index:int,*,world,entity,adventure)->str:
    if encounter.resolved:raise ValueError("encounter already resolved")
    if not 0<=choice_index<len(encounter.choices):raise ValueError("encounter choice drifted")
    choice=encounter.choices[choice_index];capacity=_capacity(entity,choice.approach,adventure);digest=hashlib.sha256(f"{encounter.encounter_id}:{choice.choice_id}:{entity.genome.semantic_sha256()}".encode()).digest();roll=int.from_bytes(digest[:8],"little")/2**64;chance=float(np.clip(.42+capacity*.62-choice.risk*.58,.08,.96));success=roll<chance;reward_kind={"grove":"biomass","mineral_vent":"rock","machine_ruin":"metal","phase_well":"crystal","spring":"water","relic_vault":"knowledge"}[encounter.kind]
    if success:
        amount=(.8+choice.reward_scale*1.8)*(1+.25*capacity);adventure.inventory[reward_kind]+=amount;adventure.inventory["knowledge"]+=.25+.45*choice.reward_scale;adventure.score+=int(20+35*choice.reward_scale);outcome=f"SUCCESS // +{amount:.2f} {reward_kind.upper()} // ANATOMY HELD"
        if encounter.kind in ("machine_ruin","phase_well","relic_vault") and choice.reward_scale>=1:
            artifact_seed=int.from_bytes(digest[8:16],"little");artifact=generate_artifact(seed=artifact_seed,provenance=encounter.encounter_id,quality=min(1,.5+.3*choice.reward_scale));adventure.artifacts.append(artifact);adventure.equip(artifact.artifact_id);outcome+=f" // RELIC {artifact.name.upper()}"
        if encounter.kind=="spring":entity.body.energy=min(1.2,entity.body.energy+.18);entity.body.heal((0,0),10,.12)
    else:
        damage=.10+.32*choice.risk;angle=math.tau*roll;point=(math.cos(angle)*5,math.sin(angle)*5);entity.body.impact(point,3.5,damage);entity.energy=max(0,entity.energy-damage*.22);outcome=f"FAILED // LOCAL ORGAN TRAUMA {damage:.2f} // FLUID LOSS"
        if encounter.kind in ("machine_ruin","phase_well","relic_vault"):
            family=4 if encounter.kind=="machine_ruin" else 3;guardian=founder_genomes(variants_per_family=1)[family];offset=np.asarray((math.cos(angle),math.sin(angle)))*2;guardian_id=world.add_organism(guardian,tuple(entity.position+offset),energy=.84);world.organisms[guardian_id].age=100;world.organisms[guardian_id].update_stage();outcome+=f" // GUARDIAN {guardian_id} AWAKENED"
    encounter.resolved=True;encounter.outcome=outcome;world.events.append({"tick":world.tick_index,"type":"site_encounter","encounter":encounter.encounter_id,"entity":entity.entity_id,"choice":choice.choice_id,"success":success,"chance":round(chance,6),"roll":round(roll,6)});return outcome
