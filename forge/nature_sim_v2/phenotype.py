from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..creature_stage_developmental import TRAITS
from .contract import ECO_TRAITS, EcoGenome


@dataclass(frozen=True, slots=True)
class PhenotypeTrait:
    key: str
    label: str
    category: str
    score: float
    description: str

    @property
    def grade(self) -> str:
        return "III" if self.score >= .82 else "II" if self.score >= .66 else "I"


def phenotype_traits(genome: EcoGenome) -> tuple[PhenotypeTrait, ...]:
    developmental = dict(zip(TRAITS, genome.developmental.traits, strict=True))
    ecology = dict(zip(ECO_TRAITS, genome.eco_traits, strict=True))
    appendage_kinds = [item.kind for item in genome.developmental.appendages]
    organ_kinds = [item.organ for item in genome.developmental.components]
    records: list[PhenotypeTrait] = []

    def add(key: str, label: str, category: str, score: float, description: str, threshold: float=.56) -> None:
        if score >= threshold:
            records.append(PhenotypeTrait(key, label, category, round(float(score), 6), description))

    add("bilateral", "bilateral chassis", "body", developmental["symmetry"], "Paired anatomy resists one-sided locomotor failure.", .50)
    add("segmented", "segmented frame", "body", developmental["segmentation"], "Serial body units localize cuts and permit graft variation.")
    add("armored", "dense shell", "body", (developmental["bone_density"]+developmental["stiffness"])*.5, "Dense structural cells blunt impact and beams.")
    add("elastic", "elastic tissue", "body", developmental["elasticity"], "Flexible connections stretch before tearing.")
    add("myomeric", "powerful myomeres", "locomotion", (developmental["muscle_density"]+developmental["muscle_strength"])*.5, "Muscle chains produce stronger grounded impulses.")
    add("gripping", "anchor grip", "locomotion", developmental["grip"], "Locomotors hold contact against the world substrate.")
    add("neural", "distributed neural mesh", "organ", developmental["neural_density"], "Behavior survives partial damage across several neural clusters.")
    add("vascular", "rich circulation", "organ", developmental["vascularity"], "Fluids distribute energy and repair material quickly.")
    add("regenerator", "regenerative", "organ", (developmental["regeneration"]+ecology["repair"])*.5, "Damaged living cells knit with persistent scars.")
    add("far_sense", "far sensing", "sense", (developmental["sensory_range"]+ecology["perception"])*.5, "Directional sensors acquire distant gradients and organisms.")
    add("phase", "phase coherent", "anomaly", developmental["phase_coherence"], "Maintains nonlocal organization and anomalous metabolism.", .62)
    add("social", "colony seeking", "ecology", (ecology["sociability"]+ecology["colony_affinity"])*.5, "Forms persistent kin colonies and settlements.")
    add("prolific", "prolific", "ecology", ecology["fertility"], "Shortens reproductive recovery when nutrition is abundant.", .60)
    add("mutable", "highly mutable", "evolution", (ecology["mutation_rate"]+ecology["mutation_scale"])*.5, "Offspring explore phenotype space more aggressively.", .52)
    add("graftable", "graft receptive", "evolution", ecology["graft_affinity"], "Accepts foreign organs and reciprocal locomotor pairs.", .58)
    if "wheel" in appendage_kinds:add("wheeled", "wheeled", "locomotion", .92, "Wheel cells roll against terrain instead of stepping.", 0)
    if "root" in appendage_kinds:add("root_drag", "root drag", "locomotion", .80, "Root bundles alternately anchor and slide.", 0)
    if sum(organ == "neural" for organ in organ_kinds) >= 2:add("multi_brain", "plural mind", "organ", .86, "Multiple neural organs provide degraded but redundant cognition.", 0)
    for mutation in genome.mutation_log[-3:]:
        records.append(PhenotypeTrait("mutation_"+mutation.replace(":","_"), mutation.replace("_"," "), "mutation", .75, "A heritable structural mutation recorded at birth."))
    records.sort(key=lambda item: (-item.score, item.category, item.key))
    return tuple(records)


def phenotype_vector(genome: EcoGenome) -> np.ndarray:
    """Compact heritable phenotype vector for future neural specialists."""
    values = np.asarray(genome.developmental.traits + genome.eco_traits + genome.diet, dtype=np.float32)
    anatomy = np.asarray((len(genome.developmental.components)/32, len(genome.developmental.appendages)/32, genome.developmental.generation/1000), dtype=np.float32)
    return np.concatenate((values, anatomy))
