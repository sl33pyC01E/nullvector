from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

from .contract import ECO_TRAITS,RESOURCE_NAMES,EcoGenome
from .state import OrganismState


@dataclass(frozen=True,slots=True)
class CohortState:
    cohort_id:str
    family:int
    lineage_id:str
    count:int
    total_energy:float
    total_reserve:float
    total_biomass:float
    mean_age:float
    mean_traits:tuple[float,...]
    generation_histogram:tuple[tuple[int,int],...]
    colony_ids:tuple[int,...]
    representative:EcoGenome
    source_entity_ids:tuple[int,...]


def demote_to_cohort(organisms:list[OrganismState],*,region_id:str)->CohortState:
    if not organisms or any(not o.alive for o in organisms):raise ValueError("LOD demotion requires living organisms")
    family=organisms[0].family;lineage=organisms[0].genome.lineage_id
    if any(o.family!=family or o.genome.lineage_id!=lineage for o in organisms):raise ValueError("LOD cohort identity mixed")
    ordered=sorted(organisms,key=lambda o:o.entity_id);count=len(ordered)
    generations={}
    for organism in ordered:generations[organism.genome.developmental.generation]=generations.get(organism.genome.developmental.generation,0)+1
    digest=hashlib.sha256((region_id+":"+lineage+":"+",".join(str(o.entity_id) for o in ordered)).encode()).hexdigest()[:16]
    return CohortState(
        f"{region_id}-{digest}",family,lineage,count,
        sum(o.energy for o in ordered),sum(o.reserve for o in ordered),
        sum(float(o.body.snapshot().alive_cells) for o in ordered),
        sum(o.age for o in ordered)/count,
        tuple(np.mean([o.genome.eco_traits for o in ordered],axis=0).tolist()),
        tuple(sorted(generations.items())),tuple(sorted({o.colony_id for o in ordered if o.colony_id is not None})),
        max(ordered,key=lambda o:(o.genome.developmental.generation,-o.entity_id)).genome,
        tuple(o.entity_id for o in ordered),
    )


def cohort_conservation(cohort:CohortState,organisms:list[OrganismState],*,tolerance:float=1e-6)->dict[str,bool]:
    return {
        "population":cohort.count==len(organisms),
        "energy":abs(cohort.total_energy-sum(o.energy for o in organisms))<=tolerance,
        "reserve":abs(cohort.total_reserve-sum(o.reserve for o in organisms))<=tolerance,
        "biomass":abs(cohort.total_biomass-sum(float(o.body.snapshot().alive_cells) for o in organisms))<=tolerance,
        "lineage":all(o.genome.lineage_id==cohort.lineage_id for o in organisms),
        "ancestry":tuple(sorted(o.entity_id for o in organisms))==tuple(sorted(cohort.source_entity_ids)),
    }


@dataclass(slots=True)
class RegionalLedger:
    region_id:str
    cohorts:dict[str,CohortState]
    resources:np.ndarray
    births:int=0
    deaths:int=0
    migrations:int=0
    speciation_events:int=0

    def __post_init__(self)->None:
        if self.resources.shape!=(len(RESOURCE_NAMES),) or np.any(self.resources<0) or not np.isfinite(self.resources).all():raise ValueError("regional resource ledger drifted")

    @property
    def population(self)->int:return sum(c.count for c in self.cohorts.values())

    @property
    def lineage_count(self)->int:return len({c.lineage_id for c in self.cohorts.values()})

