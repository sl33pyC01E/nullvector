from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib

import numpy as np

from ..creature_stage_developmental import TRAITS, DevelopmentalGenome, develop
from ..living_body_substrate import LivingBody
from .contract import ECO_TRAITS, EcoGenome


@dataclass(frozen=True, slots=True)
class EvolutionOffer:
    offer_id: str
    label: str
    description: str
    developmental: tuple[tuple[str, float], ...]
    ecological: tuple[tuple[str, float], ...]
    cost: float


TEMPLATES = (
    ("enduring_frame", "Enduring frame", "denser skeleton, stiffer load paths, longer life", (("bone_density", .10), ("stiffness", .06)), (("longevity", .08),)),
    ("contractile_limbs", "Contractile limbs", "stronger muscle chains with lower movement cost", (("muscle_density", .08), ("muscle_strength", .10), ("grip", .05)), (("move_cost", -.06),)),
    ("neural_bloom", "Neural bloom", "larger neural field and farther directional senses", (("neural_density", .10), ("sensory_range", .09)), (("perception", .09),)),
    ("vascular_repair", "Vascular repair", "faster circulation, regeneration, and wound recovery", (("vascularity", .10), ("regeneration", .09)), (("repair", .10), ("basal_metabolism", .025))),
    ("efficient_metabolism", "Efficient metabolism", "slower basal drain and broader food conversion", (("metabolism", .06),), (("basal_metabolism", -.07), ("offspring_investment", .03))),
    ("brood_instinct", "Brood instinct", "higher fertility and greater offspring investment", (("size", .02),), (("fertility", .10), ("gestation", -.05), ("offspring_investment", .08))),
    ("colony_mind", "Colony mind", "stronger cooperation, cohesion, and settlement affinity", (("neural_density", .04),), (("sociability", .10), ("colony_affinity", .11), ("cohesion", .09))),
    ("phase_adaptation", "Phase adaptation", "coherent anomaly tolerance and graft compatibility", (("phase_coherence", .12), ("elasticity", .04)), (("graft_affinity", .10), ("perception", .03))),
)


def evolution_offers(genome: EcoGenome, *, epoch: int = 0, count: int = 3) -> tuple[EvolutionOffer, ...]:
    if not 1 <= count <= 5:
        raise ValueError("evolution offer count drifted")
    digest = hashlib.sha256(f"{genome.semantic_sha256()}:{int(epoch)}".encode()).digest()
    start = int.from_bytes(digest[:4], "little") % len(TEMPLATES)
    stride = 1 + int.from_bytes(digest[4:8], "little") % (len(TEMPLATES)-1)
    while np.gcd(stride, len(TEMPLATES)) != 1:
        stride += 1
    selected = []
    cursor = start
    while len(selected) < count:
        template = TEMPLATES[cursor % len(TEMPLATES)]
        if template not in selected:
            selected.append(template)
        cursor += stride
    cost = 1.0 + genome.developmental.generation * .06
    return tuple(EvolutionOffer(f"evo-{name}-{digest[index]:02x}", label, description, developmental, ecological, cost) for index, (name, label, description, developmental, ecological) in enumerate(selected))


def apply_offer(genome: EcoGenome, offer: EvolutionOffer, *, seed: int) -> EcoGenome:
    developmental = np.asarray(genome.developmental.traits, np.float64)
    ecology = np.asarray(genome.eco_traits, np.float64)
    for name, amount in offer.developmental:
        developmental[TRAITS.index(name)] = np.clip(developmental[TRAITS.index(name)] + amount, 0, 1)
    for name, amount in offer.ecological:
        ecology[ECO_TRAITS.index(name)] = np.clip(ecology[ECO_TRAITS.index(name)] + amount, 0, 1)
    old = genome.developmental
    identity = hashlib.sha256(f"{old.genome_id}:{offer.offer_id}:{int(seed)}".encode()).hexdigest()[:14]
    evolved = DevelopmentalGenome(f"directed_g{old.generation+1}_{identity}", int(seed), old.family_mix, tuple(developmental), old.components, old.appendages, old.generation+1, (old.genome_id,))
    develop(evolved)
    return EcoGenome(evolved, tuple(ecology), genome.diet, genome.lineage_id, genome.mutation_log+(f"directed:{offer.offer_id}",))


def _transfer_body(old: LivingBody, new: LivingBody) -> None:
    old_xy = old.organism.cell_xy.astype(np.float64)
    old_tissue = old.organism.tissue
    for index, (point, tissue) in enumerate(zip(new.organism.cell_xy, new.organism.tissue)):
        candidates = np.flatnonzero(old_tissue == tissue)
        if not len(candidates):
            candidates = np.arange(len(old_xy))
        nearest = int(candidates[np.argmin(np.sum((old_xy[candidates]-point)**2, axis=1))])
        new.health[index] = old.health[nearest]
        new.scar[index] = old.scar[nearest]
        new.fluid[index] = old.fluid[nearest]
        new.separation_age[index] = old.separation_age[nearest]
    new.energy = old.energy
    new.tick_index = old.tick_index
    new.incapacitated = old.incapacitated
    new.dead = old.dead
    new.external_puddle = list(old.external_puddle)
    new.polyps = list(old.polyps)
    new.biomass = list(old.biomass)


def metamorphose(entity, offer: EvolutionOffer, *, seed: int) -> EcoGenome:
    evolved = apply_offer(entity.genome, offer, seed=seed)
    body = LivingBody(develop(evolved.developmental), seed=evolved.developmental.seed)
    _transfer_body(entity.body, body)
    entity.genome = evolved
    entity.body = body
    entity.neural_contacts = np.zeros(len(body.organism.genome.appendages), dtype=np.bool_)
    entity.neural_muscles = np.zeros(len(body.organism.muscles), dtype=np.float32)
    entity.reproduction_cooldown = max(entity.reproduction_cooldown, 12.0)
    entity.energy = max(.05, entity.energy-.12)
    entity.reserve = max(0, entity.reserve-.08)
    return evolved
