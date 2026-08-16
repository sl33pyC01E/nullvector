from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np

from .phenotype import phenotype_traits, phenotype_vector


@dataclass(slots=True)
class CladeRecord:
    clade_id: str
    lineage_id: str
    family: int
    signature: tuple[int, ...]
    trait_labels: tuple[str, ...]
    population: int = 0
    births: int = 0
    deaths: int = 0
    max_generation: int = 0
    mean_energy: float = 0.0
    mean_integrity: float = 0.0
    territory: float = 0.0

    @property
    def fitness(self) -> float:
        return (self.births+1)/(self.deaths+1) * math.sqrt(self.population+1) * (.4+.6*self.mean_integrity)


class EvolutionLedger:
    """Online natural-selection ledger over heritable phenotype clusters."""
    def __init__(self) -> None:
        self.clades: dict[str, CladeRecord] = {}
        self.entity_clade: dict[int, str] = {}
        self.ancestry: set[tuple[str, str]] = set()
        self.last_event = 0
        self.observations = 0

    @staticmethod
    def signature(genome) -> tuple[int, ...]:
        vector = phenotype_vector(genome)
        # Coarse bins represent reproductively meaningful ecotypes rather than
        # assigning a new species to every tiny mutation.
        sampled = vector[np.asarray((1,2,3,4,5,6,7,8,11,12,13,14,18,19,20,21,24,25,27,28,31,41,42))]
        return tuple(np.clip(np.rint(sampled*5),0,5).astype(np.uint8).tolist())

    @classmethod
    def clade_id(cls, genome) -> str:
        signature = cls.signature(genome)
        payload = f"{genome.lineage_id}:{genome.family}:"+",".join(map(str,signature))
        return "clade-"+hashlib.sha256(payload.encode()).hexdigest()[:12]

    def observe(self, world) -> None:
        self.observations += 1
        living = [entity for entity in world.organisms.values() if entity.alive]
        groups: dict[str, list] = {}
        for entity in living:
            clade_id = self.clade_id(entity.genome);groups.setdefault(clade_id,[]).append(entity);self.entity_clade[entity.entity_id]=clade_id
            if clade_id not in self.clades:
                labels=tuple(item.label for item in phenotype_traits(entity.genome)[:5]);self.clades[clade_id]=CladeRecord(clade_id,entity.genome.lineage_id,entity.family,self.signature(entity.genome),labels)
        for record in self.clades.values():record.population=0
        for clade_id,members in groups.items():
            record=self.clades[clade_id];record.population=len(members);record.max_generation=max(record.max_generation,max(item.genome.developmental.generation for item in members));record.mean_energy=float(np.mean([item.energy for item in members]));record.mean_integrity=float(np.mean([item.body.systems()["integrity"] for item in members]));positions=np.asarray([item.position for item in members]);record.territory=float(np.ptp(positions,axis=0).prod()) if len(members)>1 else 0.0
        for event in world.events[self.last_event:]:
            if event.get("type") in ("birth","vegetative_spread","polyp"):
                child=world.organisms.get(int(event["entity"]));
                if child is not None:
                    child_clade=self.clade_id(child.genome);self.entity_clade[child.entity_id]=child_clade
                    if child_clade not in self.clades:self.clades[child_clade]=CladeRecord(child_clade,child.genome.lineage_id,child.family,self.signature(child.genome),tuple(item.label for item in phenotype_traits(child.genome)[:5]))
                    self.clades[child_clade].births+=1
                    for parent_id in child.parent_ids:
                        parent_clade=self.entity_clade.get(parent_id)
                        if parent_clade and parent_clade!=child_clade:self.ancestry.add((parent_clade,child_clade))
            elif event.get("type")=="death":
                clade_id=self.entity_clade.get(int(event["entity"]));
                if clade_id in self.clades:self.clades[clade_id].deaths+=1
        self.last_event=len(world.events)

    @property
    def diversity(self) -> float:
        counts=np.asarray([record.population for record in self.clades.values() if record.population>0],dtype=np.float64)
        if not len(counts):return 0.0
        probabilities=counts/counts.sum();return float(-(probabilities*np.log2(probabilities)).sum())

    def dominant(self,limit:int=5)->tuple[CladeRecord,...]:
        return tuple(sorted((record for record in self.clades.values() if record.population),key=lambda record:(-record.fitness,-record.population,record.clade_id))[:limit])
