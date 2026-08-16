from __future__ import annotations

from dataclasses import dataclass,field
import math


FORMAT="nullvector-qud-society-scaffold/1.0.0"
CULTURAL_TRAITS=("collectivism","curiosity","militancy","mercy","ritual","industry","biophilia","anomaly_affinity")
TECHNOLOGIES=("cell_cultivation","organ_grafting","mineral_tools","ballistics","phase_lensing","neural_weaving","architecture","agriculture","medicine","machine_logic")
ACTIVITIES=("forage","hunt","heal","breed","graft","craft","build","trade","explore","map","study_anomaly","defend","negotiate","raid","found_colony","recover_relic")


@dataclass(frozen=True,slots=True)
class BuildingPlan:
    building_id:str
    purpose:str
    origin:tuple[int,int]
    width:int
    height:int
    cells:tuple[tuple[int,int,str],...]
    entrances:tuple[tuple[int,int],...]

    def __post_init__(self)->None:
        if not 7<=self.width<=41 or not 7<=self.height<=41 or not self.cells or not self.entrances:raise ValueError("society building geometry drifted")
        if any(material not in {"wall","floor","door","utility","garden","storage"} for _,_,material in self.cells):raise ValueError("society building material drifted")


@dataclass(slots=True)
class SettlementState:
    settlement_id:str
    faction_id:str
    center:tuple[float,float]
    population:int
    wealth:float
    food:float
    power:float
    buildings:list[BuildingPlan]=field(default_factory=list)
    roads:set[tuple[int,int]]=field(default_factory=set)
    founded_tick:int=0
    stockpiles:dict[str,float]=field(default_factory=dict)
    production:dict[str,float]=field(default_factory=dict)
    shortages:int=0
    projects_completed:int=0


@dataclass(slots=True)
class FactionState:
    faction_id:str
    name:str
    family:int
    lineage_id:str
    cultural_traits:tuple[float,...]
    technologies:set[str]
    settlement_ids:set[str]
    relations:dict[str,float]
    knowledge:float=.1
    cohesion:float=.6
    doctrine:str="survive"

    def __post_init__(self)->None:
        if len(self.cultural_traits)!=len(CULTURAL_TRAITS) or any(not math.isfinite(v) or not 0<=v<=1 for v in self.cultural_traits):raise ValueError("society cultural traits drifted")


@dataclass(frozen=True,slots=True)
class HistoryEvent:
    tick:int
    kind:str
    actors:tuple[str,...]
    location:tuple[float,float]
    description:str
    consequences:tuple[tuple[str,float],...]


@dataclass(frozen=True,slots=True)
class Activity:
    activity_id:str
    kind:str
    issuer:str
    location:tuple[float,float]
    difficulty:float
    reward_materials:tuple[tuple[str,float],...]
    description:str
