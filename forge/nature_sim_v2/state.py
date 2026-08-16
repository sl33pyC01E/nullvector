from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from ..creature_stage_developmental import develop
from ..living_body_substrate import LivingBody
from .contract import EcoGenome, LIFE_STAGES, RESOURCE_NAMES


@dataclass(slots=True)
class OrganismState:
    entity_id: int
    genome: EcoGenome
    body: LivingBody
    position: np.ndarray
    velocity: np.ndarray
    age: float = 0.0
    energy: float = .55
    reserve: float = .25
    stage: str = "juvenile"
    intent: str = "rest"
    target: np.ndarray | None = None
    reproduction_cooldown: float = 0.0
    gestation_remaining: float = 0.0
    mate_id: int | None = None
    colony_id: int | None = None
    alive: bool = True
    decomposition: float = 0.0
    heading: float = 0.0
    birth_tick: int = 0
    parent_ids: tuple[int, ...] = ()
    consumed: np.ndarray = field(default_factory=lambda: np.zeros(len(RESOURCE_NAMES), dtype=np.float64))
    neural_contacts: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.bool_))
    neural_muscles: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    polyp_cursor: int = 0

    @classmethod
    def spawn(cls, entity_id: int, genome: EcoGenome, position: tuple[float,float], *, birth_tick: int = 0, parent_ids: tuple[int,...] = (), energy: float = .55) -> "OrganismState":
        body = LivingBody(develop(genome.developmental), seed=genome.developmental.seed)
        body.energy = min(1.0, max(.15, energy))
        return cls(entity_id, genome, body, np.asarray(position,dtype=np.float64), np.zeros(2,dtype=np.float64), energy=energy, reserve=energy*.35, birth_tick=birth_tick, parent_ids=parent_ids)

    @property
    def family(self) -> int:
        return self.genome.family

    def update_stage(self) -> None:
        if not self.alive:
            self.stage = "decomposed" if self.decomposition >= 1 else "dead"
            return
        maturity = 12 + self.genome.trait("maturity") * 38
        longevity = 90 + self.genome.trait("longevity") * 310
        if self.age < maturity * .18:
            self.stage = "embryo"
        elif self.age < maturity:
            self.stage = "juvenile"
        elif self.age < longevity * .78:
            self.stage = "mature"
        else:
            self.stage = "senescent"
        if self.stage not in LIFE_STAGES:
            raise AssertionError("life stage drifted")

    def finite(self) -> bool:
        return bool(np.isfinite(self.position).all() and np.isfinite(self.velocity).all() and all(math.isfinite(v) for v in (self.age,self.energy,self.reserve,self.decomposition)))


@dataclass(slots=True)
class ColonyState:
    colony_id: int
    family: int
    founder_lineage: str
    member_ids: set[int]
    center: np.ndarray
    generation: int = 0
    fissions: int = 0
