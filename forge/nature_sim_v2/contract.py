from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

from ..creature_stage_developmental import DevelopmentalGenome, FAMILIES


FORMAT = "nullvector-nature-sim-v2/1.0.0"
RESOURCE_NAMES = (
    "water", "light", "mineral", "charge", "phase", "oxygen", "heat",
    "toxin", "flora", "biomass",
)
ECO_TRAITS = (
    "maturity", "longevity", "gestation", "fertility", "mutation_rate",
    "mutation_scale", "basal_metabolism", "move_cost", "repair",
    "aggression", "sociability", "perception", "offspring_investment",
    "colony_affinity", "cohesion", "graft_affinity",
)
LIFE_STAGES = ("embryo", "juvenile", "mature", "senescent", "dead", "decomposed")
INTENTS = (
    "rest", "forage", "hunt", "flee", "mate", "follow", "photosynthesize",
    "mine", "phase_feed", "repair", "guard", "explore",
)


@dataclass(frozen=True, slots=True)
class EcoGenome:
    developmental: DevelopmentalGenome
    eco_traits: tuple[float, ...]
    diet: tuple[float, ...]
    lineage_id: str
    mutation_log: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(self.eco_traits) != len(ECO_TRAITS) or not all(math.isfinite(v) and 0 <= v <= 1 for v in self.eco_traits):
            raise ValueError("nature ecology traits drifted")
        if len(self.diet) != len(RESOURCE_NAMES) or not all(math.isfinite(v) and 0 <= v <= 1 for v in self.diet):
            raise ValueError("nature diet drifted")
        if sum(self.diet) <= 0 or not self.lineage_id:
            raise ValueError("nature lineage/diet is empty")

    @property
    def family(self) -> int:
        return max(range(len(FAMILIES)), key=self.developmental.family_mix.__getitem__)

    def trait(self, name: str) -> float:
        return self.eco_traits[ECO_TRAITS.index(name)]

    def resource(self, name: str) -> float:
        return self.diet[RESOURCE_NAMES.index(name)]

    def semantic_sha256(self) -> str:
        payload = {
            "developmental_id": self.developmental.genome_id,
            "seed": self.developmental.seed,
            "generation": self.developmental.generation,
            "parents": self.developmental.parent_ids,
            "family_mix": self.developmental.family_mix,
            "traits": self.developmental.traits,
            "components": [
                (c.component_id, c.kind, c.anchor, c.radius, c.parent, c.side, c.trait_delta, c.organ)
                for c in self.developmental.components
            ],
            "appendages": [
                (a.appendage_id, a.kind, a.root_component, a.root_offset, a.endpoint,
                 a.segments, a.side, a.phase, a.bend, a.paired_with, a.trait_delta)
                for a in self.developmental.appendages
            ],
            "eco_traits": self.eco_traits,
            "diet": self.diet,
            "lineage_id": self.lineage_id,
            "mutation_log": self.mutation_log,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class WorldSnapshot:
    tick: int
    time: float
    population: int
    births: int
    deaths: int
    predation_events: int
    colony_count: int
    lineage_count: int
    family_counts: tuple[int, ...]
    resource_totals: tuple[float, ...]
    mutation_count: int
    semantic_sha256: str

