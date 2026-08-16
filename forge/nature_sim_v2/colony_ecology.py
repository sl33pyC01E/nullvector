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
        self.role_policy = None

    @staticmethod
    def choose_role(entity) -> str:
        traits=entity.genome.developmental.traits;eco=entity.genome.eco_traits
        scores=np.asarray((
            .15+.4*(entity.genome.diet[8]+entity.genome.diet[9])/2,
            .15+.4*(traits[13]+eco[11])/2,
            .15+.4*(traits[3]+traits[5]+eco[9])/3,
            .15+.4*(traits[11]+eco[8]+traits[9])/3,
            .15+.4*(eco[3]+eco[12])/2,
            .15+.4*(traits[3]+traits[6]+eco[15])/3,
        ),dtype=np.float64)
        # Identity phase prevents genetically similar colonies from assigning
        # every member to the same profession.
        scores[(entity.entity_id+entity.genome.developmental.seed)%len(ROLES)]+=.58
        return ROLES[int(np.argmax(scores))]

    def step(self,world,delta:float)->None:
        active=set(world.colonies)
        for colony_id in list(self.states):
            if colony_id not in active:del self.states[colony_id]
        for colony_id,colony in sorted(world.colonies.items()):
            state=self.states.setdefault(colony_id,ColonyEcologyState(colony_id));members=[world.organisms[i] for i in sorted(colony.member_ids) if i in world.organisms and world.organisms[i].alive]
            if not members:continue
            neural_assignments=self.role_policy.assign(members,state) if self.role_policy is not None else None
            for member in members:state.assignments[member.entity_id]=neural_assignments.get(member.entity_id,self.choose_role(member)) if neural_assignments is not None else self.choose_role(member)
            state.assignments={entity_id:role for entity_id,role in state.assignments.items() if entity_id in colony.member_ids}
            distances=[float(np.linalg.norm(world._delta(member.position,colony.center))) for member in members];state.cohesion=float(math.exp(-np.mean(distances)/6))
            # Only surplus enters the commons; total organism + store energy is
            # conserved except for a small transfer cost.
            for member in members:
                surplus=max(0,member.energy-.88);neural_action=self.role_policy.action(member.entity_id) if self.role_policy is not None else None;rate=.006*(.4+member.genome.trait("sociability")) if neural_action is None else max(.001,float(neural_action[0])*.08);donation=min(surplus,delta*rate)
                member.energy-=donation;state.energy_store+=donation*.985
            receivers=sorted((member for member in members if member.energy<.34),key=lambda member:(member.energy,member.entity_id))
            for member in receivers:
                amount=min(state.energy_store,.34-member.energy,delta*.009);state.energy_store-=amount;member.energy+=amount;state.transfers+=int(amount>0)
            medics=[member for member in members if state.assignments.get(member.entity_id)=="medic" and member.energy>.30]
            wounded=sorted((member for member in members if member.body.systems()["integrity"]<.92),key=lambda member:(-(self.role_policy.action(member.entity_id) or (0,1-member.body.systems()["integrity"],0))[1] if self.role_policy is not None else member.body.systems()["integrity"],member.entity_id))
            for medic,target in zip(medics,wounded):
                budget=min(.004*delta,medic.energy-.28,state.energy_store+.002)
                if budget<=0:continue
                from_store=min(state.energy_store,budget*.5);state.energy_store-=from_store;medic.energy-=budget-from_store;target.body.heal((0,0),10,budget*7);state.repairs+=1

    def assignment(self,entity_id:int)->str|None:
        return next((state.assignments[entity_id] for state in self.states.values() if entity_id in state.assignments),None)
