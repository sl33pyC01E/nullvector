from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np


ROLES = ("gatherer", "scout", "defender", "medic", "breeder", "builder")


@dataclass(slots=True)
class ColonyEcologyState:
    colony_id: int
    assignments: dict[int, str] = field(default_factory=dict)
    energy_store: float = 0.0
    material_store: np.ndarray = field(default_factory=lambda: np.zeros(10, dtype=np.float64))
    cohesion: float = 1.0
    transfers: int = 0
    repairs: int = 0


class ColonyEcology:
    """Physical resource sharing and role specialization for kin colonies."""
    def __init__(self) -> None:
        self.states: dict[int, ColonyEcologyState] = {}

    @staticmethod
    def choose_role(entity) -> str:
        traits=entity.genome.developmental.traits;eco=entity.genome.eco_traits
        scores=np.asarray((
            .30+entity.genome.diet[8]+entity.genome.diet[9],
            .25+traits[13]+eco[11],
            .20+traits[3]+traits[5]+eco[9],
            .18+traits[11]+eco[8]+traits[9],
            .12+eco[3]+eco[12],
            .15+traits[3]+traits[6]+eco[15],
        ),dtype=np.float64)
        # Identity phase prevents genetically similar colonies from assigning
        # every member to the same profession.
        scores[(entity.entity_id+entity.genome.developmental.seed)%len(ROLES)]+=.23
        return ROLES[int(np.argmax(scores))]

    def step(self,world,delta:float)->None:
        active=set(world.colonies)
        for colony_id in list(self.states):
            if colony_id not in active:del self.states[colony_id]
        for colony_id,colony in sorted(world.colonies.items()):
            state=self.states.setdefault(colony_id,ColonyEcologyState(colony_id));members=[world.organisms[i] for i in sorted(colony.member_ids) if i in world.organisms and world.organisms[i].alive]
            if not members:continue
            for member in members:state.assignments[member.entity_id]=self.choose_role(member)
            state.assignments={entity_id:role for entity_id,role in state.assignments.items() if entity_id in colony.member_ids}
            distances=[float(np.linalg.norm(world._delta(member.position,colony.center))) for member in members];state.cohesion=float(math.exp(-np.mean(distances)/6))
            # Only surplus enters the commons; total organism + store energy is
            # conserved except for a small transfer cost.
            for member in members:
                surplus=max(0,member.energy-.88);donation=min(surplus,delta*.006*(.4+member.genome.trait("sociability")))
                member.energy-=donation;state.energy_store+=donation*.985
            receivers=sorted((member for member in members if member.energy<.34),key=lambda member:(member.energy,member.entity_id))
            for member in receivers:
                amount=min(state.energy_store,.34-member.energy,delta*.009);state.energy_store-=amount;member.energy+=amount;state.transfers+=int(amount>0)
            medics=[member for member in members if state.assignments.get(member.entity_id)=="medic" and member.energy>.30]
            wounded=sorted((member for member in members if member.body.systems()["integrity"]<.92),key=lambda member:(member.body.systems()["integrity"],member.entity_id))
            for medic,target in zip(medics,wounded):
                budget=min(.004*delta,medic.energy-.28,state.energy_store+.002)
                if budget<=0:continue
                from_store=min(state.energy_store,budget*.5);state.energy_store-=from_store;medic.energy-=budget-from_store;target.body.heal((0,0),10,budget*7);state.repairs+=1

    def assignment(self,entity_id:int)->str|None:
        return next((state.assignments[entity_id] for state in self.states.values() if entity_id in state.assignments),None)
