from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib

import numpy as np

from ..creature_stage_developmental import AppendageGene,ComponentGene,TRAITS, DevelopmentalGenome, develop
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
    structural: str | None = None


TEMPLATES = (
    ("enduring_frame", "Enduring frame", "denser skeleton, stiffer load paths, longer life", (("bone_density", .10), ("stiffness", .06)), (("longevity", .08),), None),
    ("contractile_limbs", "Contractile limbs", "stronger muscle chains with lower movement cost", (("muscle_density", .08), ("muscle_strength", .10), ("grip", .05)), (("move_cost", -.06),), None),
    ("neural_bloom", "Neural bloom", "larger neural field and farther directional senses", (("neural_density", .10), ("sensory_range", .09)), (("perception", .09),), None),
    ("vascular_repair", "Vascular repair", "faster circulation, regeneration, and wound recovery", (("vascularity", .10), ("regeneration", .09)), (("repair", .10), ("basal_metabolism", .025)), None),
    ("efficient_metabolism", "Efficient metabolism", "slower basal drain and broader food conversion", (("metabolism", .06),), (("basal_metabolism", -.07), ("offspring_investment", .03)), None),
    ("brood_instinct", "Brood instinct", "higher fertility and greater offspring investment", (("size", .02),), (("fertility", .10), ("gestation", -.05), ("offspring_investment", .08)), None),
    ("colony_mind", "Colony mind", "stronger cooperation, cohesion, and settlement affinity", (("neural_density", .04),), (("sociability", .10), ("colony_affinity", .11), ("cohesion", .09)), None),
    ("phase_adaptation", "Phase adaptation", "coherent anomaly tolerance and graft compatibility", (("phase_coherence", .12), ("elasticity", .04)), (("graft_affinity", .10), ("perception", .03)), None),
    ("armor_lobes", "Paired armor lobes", "grows two load-bearing lateral armor organs", (("bone_density", .05),), (("cohesion", .03),), "armor_lobes"),
    ("sensor_crown", "Sensor crown", "grows a directed sensory organ above the chassis", (("sensory_range", .08),), (("perception", .08),), "sensor_crown"),
    ("storage_lobes", "Storage lobes", "grows symmetric reserves for famine and gestation", (("metabolism", .04),), (("offspring_investment", .06),), "storage_lobes"),
    ("neural_lobes", "Neural lobes", "grows redundant paired behavior tissue", (("neural_density", .08),), (("sociability", .04),), "neural_lobes"),
    ("hardpoint_pair", "Hardpoint pair", "grows reciprocal weapon or tool mounts", (("stiffness", .04), ("grip", .05)), (("aggression", .04),), "hardpoint_pair"),
    ("locomotor_pair", "Locomotor pair", "grows a new symmetric grounded limb pair", (("muscle_strength", .07),), (("move_cost", -.03),), "locomotor_pair"),
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
    return tuple(EvolutionOffer(f"evo-{name}-{digest[index]:02x}", label, description, developmental, ecological, cost, structural) for index, (name, label, description, developmental, ecological, structural) in enumerate(selected))


def _grow_structure(old:DevelopmentalGenome,kind:str,identity:str)->tuple[tuple[ComponentGene,...],tuple[AppendageGene,...]]:
    components=list(old.components);appendages=list(old.appendages);root=next((item for item in components if item.kind=="soma"),components[0]);rx,ry=root.radius;ax,ay=root.anchor
    def component(suffix,component_kind,anchor,radius,side,organ):
        components.append(ComponentGene(f"mut_{identity}_{suffix}",component_kind,anchor,radius,root.component_id,side,(0.0,)*len(TRAITS),organ))
    if kind in ("armor_lobes","storage_lobes","neural_lobes"):
        component_kind,organ,radius={"armor_lobes":("armor","armor",(max(.8,rx*.34),max(.8,ry*.48))),"storage_lobes":("storage","storage",(max(.8,rx*.30),max(.8,ry*.38))),"neural_lobes":("neural_cluster","brain",(max(.8,rx*.26),max(.8,ry*.30)))}[kind]
        for side in (-1,1):component(f"{kind}_{side:+d}",component_kind,(ax+side*(rx+radius[0]*.55),ay-.08*ry),radius,side,organ)
    elif kind=="sensor_crown":component(kind,"sensor_crown",(ax,ay-ry*1.15),(max(.8,rx*.42),max(.8,ry*.27)),0,"sensor")
    elif kind in ("hardpoint_pair","locomotor_pair"):
        appendage_kind="hardpoint" if kind=="hardpoint_pair" else "leg";left=f"mut_{identity}_{kind}_l";right=f"mut_{identity}_{kind}_r";y=ry*.35 if appendage_kind=="hardpoint" else ry*.72;end_y=ay+ry*.78 if appendage_kind=="hardpoint" else ay+ry*2.0
        appendages.extend((AppendageGene(left,appendage_kind,root.component_id,(-rx*.72,y),(ax-rx*1.45,end_y),3,-1,0.0,1,right,(0.0,)*len(TRAITS)),AppendageGene(right,appendage_kind,root.component_id,(rx*.72,y),(ax+rx*1.45,end_y),3,1,.5,-1,left,(0.0,)*len(TRAITS))))
    return tuple(components),tuple(appendages)


def apply_offer(genome: EcoGenome, offer: EvolutionOffer, *, seed: int) -> EcoGenome:
    developmental = np.asarray(genome.developmental.traits, np.float64)
    ecology = np.asarray(genome.eco_traits, np.float64)
    for name, amount in offer.developmental:
        developmental[TRAITS.index(name)] = np.clip(developmental[TRAITS.index(name)] + amount, 0, 1)
    for name, amount in offer.ecological:
        ecology[ECO_TRAITS.index(name)] = np.clip(ecology[ECO_TRAITS.index(name)] + amount, 0, 1)
    old = genome.developmental
    identity = hashlib.sha256(f"{old.genome_id}:{offer.offer_id}:{int(seed)}".encode()).hexdigest()[:14]
    components,appendages=(old.components,old.appendages) if offer.structural is None else _grow_structure(old,offer.structural,identity)
    evolved = DevelopmentalGenome(f"directed_g{old.generation+1}_{identity}", int(seed), old.family_mix, tuple(developmental), components, appendages, old.generation+1, (old.genome_id,))
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
