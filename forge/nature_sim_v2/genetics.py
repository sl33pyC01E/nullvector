from __future__ import annotations

from dataclasses import replace
import hashlib

import numpy as np

from ..creature_stage_developmental import AppendageGene, ComponentGene, DevelopmentalGenome, FAMILIES, TRAITS, develop
from ..creature_stage_morphology_v2 import morphology_review_genomes
from .contract import ECO_TRAITS, RESOURCE_NAMES, EcoGenome


FAMILY_DIETS = (
    # water light mineral charge phase oxygen heat toxin flora biomass
    (.35, .05, .32, .18, .02, .12, .04, .01, .48, .82),
    (.40, .02, .08, .01, .00, .22, .02, .00, .78, .92),
    (.86, .94, .58, .02, .01, .28, .08, .00, .04, .18),
    (.06, .08, .32, .22, .98, .02, .74, .24, .06, .18),
    (.05, .00, .92, .98, .04, .05, .38, .08, .00, .12),
)


FAMILY_ECO = (
    (.24,.72,.32,.40,.17,.16,.48,.34,.52,.58,.74,.70,.62,.70,.76,.42),
    (.18,.55,.24,.58,.22,.18,.56,.42,.45,.62,.68,.78,.48,.62,.72,.34),
    (.12,.78,.18,.76,.28,.20,.22,.05,.78,.08,.86,.52,.34,.94,.88,.38),
    (.31,.88,.42,.24,.34,.30,.18,.14,.66,.46,.38,.92,.58,.72,.54,.84),
    (.35,.94,.30,.20,.16,.12,.24,.26,.42,.40,.82,.84,.72,.88,.94,.66),
)


def founder_genomes(*, variants_per_family: int = 3) -> tuple[EcoGenome, ...]:
    if not 1 <= variants_per_family <= 6:
        raise ValueError("founder variant count drifted")
    priors = morphology_review_genomes()
    result: list[EcoGenome] = []
    for family in range(len(FAMILIES)):
        for variant in range(variants_per_family):
            developmental = priors[family * 6 + variant]
            lineage = f"{FAMILIES[family]}-{hashlib.sha256(developmental.genome_id.encode()).hexdigest()[:10]}"
            result.append(EcoGenome(developmental, FAMILY_ECO[family], FAMILY_DIETS[family], lineage))
    return tuple(result)


def _blend_tuple(left: tuple[float, ...], right: tuple[float, ...], alpha: float, noise: np.ndarray, scale: float) -> tuple[float, ...]:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    return tuple(np.clip(a * (1 - alpha) + b * alpha + noise * scale, 0, 1).tolist())


def _blend_component(primary: ComponentGene, other: ComponentGene, alpha: float, rng: np.random.Generator, scale: float) -> ComponentGene:
    anchor = np.asarray(primary.anchor) * (1-alpha) + np.asarray(other.anchor) * alpha
    radius = np.asarray(primary.radius) * (1-alpha) + np.asarray(other.radius) * alpha
    anchor += rng.normal(0, scale * .35, 2)
    radius *= np.exp(rng.normal(0, scale * .18, 2))
    radius = np.clip(radius, .65, 11.8)
    delta = _blend_tuple(primary.trait_delta, other.trait_delta, alpha, rng.normal(0, 1, len(TRAITS)), scale * .08)
    # Trait deltas may be negative; restore a bounded signed interpolation.
    delta = tuple(np.clip(np.asarray(primary.trait_delta)*(1-alpha)+np.asarray(other.trait_delta)*alpha+rng.normal(0,scale*.025,len(TRAITS)), -.45, .45))
    return replace(primary, anchor=tuple(anchor), radius=tuple(radius), trait_delta=delta, organ=primary.organ if rng.random() < .75 else other.organ)


def _mutate_appendage_pair(appendages: list[AppendageGene], index: int, rng: np.random.Generator, scale: float) -> str:
    item = appendages[index]
    partner_index = next((i for i,a in enumerate(appendages) if a.appendage_id == item.paired_with), None)
    factor_x = float(np.exp(rng.normal(0, scale * .28)))
    factor_y = float(np.exp(rng.normal(0, scale * .28)))
    for target_index in ([index] if partner_index is None else [index, partner_index]):
        target = appendages[target_index]
        side = target.side
        endpoint = (abs(target.endpoint[0]) * factor_x * (side if side else 1), target.endpoint[1] * factor_y)
        root = (abs(target.root_offset[0]) * factor_x * (side if side else 1), target.root_offset[1] * factor_y)
        appendages[target_index] = replace(target, endpoint=endpoint, root_offset=root)
    return f"paired_appendage_scale:{item.kind}"


def recombine(left: EcoGenome, right: EcoGenome, *, seed: int, allow_graft: bool = False) -> EcoGenome:
    rng = np.random.default_rng(seed)
    primary, secondary = (left, right) if rng.random() < .5 else (right, left)
    same_family = left.family == right.family
    if not same_family and not allow_graft:
        secondary = primary
    mutation_rate = (left.trait("mutation_rate") + right.trait("mutation_rate")) * .5
    mutation_scale = (left.trait("mutation_scale") + right.trait("mutation_scale")) * .5
    alpha = float(rng.uniform(.35, .65))
    global_traits = _blend_tuple(
        primary.developmental.traits, secondary.developmental.traits, alpha,
        rng.normal(0, 1, len(TRAITS)), mutation_scale * .035,
    )
    secondary_components = {c.component_id: c for c in secondary.developmental.components}
    components = [
        _blend_component(c, secondary_components.get(c.component_id, c), alpha, rng, mutation_scale)
        for c in primary.developmental.components
    ]
    secondary_appendages = {a.appendage_id: a for a in secondary.developmental.appendages}
    appendages: list[AppendageGene] = []
    for item in primary.developmental.appendages:
        other = secondary_appendages.get(item.appendage_id, item)
        endpoint = np.asarray(item.endpoint)*(1-alpha) + np.asarray(other.endpoint)*alpha
        root = np.asarray(item.root_offset)*(1-alpha) + np.asarray(other.root_offset)*alpha
        appendages.append(replace(item, endpoint=tuple(endpoint), root_offset=tuple(root), segments=item.segments if rng.random()<.65 else other.segments))
    mutations: list[str] = []
    if rng.random() < mutation_rate and components:
        target = int(rng.integers(len(components)))
        component = components[target]
        factor = float(np.exp(rng.normal(0, mutation_scale * .22)))
        components[target] = replace(component, radius=tuple(np.clip(np.asarray(component.radius)*factor,.65,11.8)))
        mutations.append(f"organ_scale:{component.component_id}")
    if rng.random() < mutation_rate and appendages:
        mutations.append(_mutate_appendage_pair(appendages, int(rng.integers(len(appendages))), rng, mutation_scale))
    # Cross-family grafting is rare and always copies a complete reciprocal pair.
    if allow_graft and left.family != right.family and rng.random() < min(.12, (left.trait("graft_affinity") + right.trait("graft_affinity")) * .07):
        candidates = [a for a in secondary.developmental.appendages if a.paired_with is not None]
        if candidates:
            source = candidates[int(rng.integers(len(candidates)))]
            partner = next(a for a in secondary.developmental.appendages if a.appendage_id == source.paired_with)
            if source.root_component in {c.component_id for c in components}:
                suffix = hashlib.sha256(f"{seed}:{source.appendage_id}".encode()).hexdigest()[:5]
                left_id, right_id = f"graft_{suffix}_l", f"graft_{suffix}_r"
                pair = (source, partner) if source.side < partner.side else (partner, source)
                appendages.extend((replace(pair[0], appendage_id=left_id, paired_with=right_id), replace(pair[1], appendage_id=right_id, paired_with=left_id)))
                mutations.append(f"locomotor_graft:{source.kind}")
    eco = _blend_tuple(left.eco_traits, right.eco_traits, alpha, rng.normal(0,1,len(ECO_TRAITS)), mutation_scale*.025)
    diet = _blend_tuple(left.diet, right.diet, alpha, rng.normal(0,1,len(RESOURCE_NAMES)), mutation_scale*.018)
    generation = max(left.developmental.generation, right.developmental.generation) + 1
    genome_id = f"eco_g{generation}_{seed:016x}"
    family_mix = _blend_tuple(left.developmental.family_mix, right.developmental.family_mix, alpha, np.zeros(len(FAMILIES)), 0)
    family_mix_array = np.asarray(family_mix)
    family_mix = tuple((family_mix_array / family_mix_array.sum()).tolist())
    developmental = DevelopmentalGenome(
        genome_id, int(seed), family_mix, global_traits, tuple(components), tuple(appendages),
        generation, (left.developmental.genome_id, right.developmental.genome_id),
    )
    # Development is the structural acceptance gate. Invalid mutations never enter nature.
    develop(developmental)
    lineage = left.lineage_id if same_family else f"hybrid-{left.lineage_id[:8]}-{right.lineage_id[:8]}"
    return EcoGenome(developmental, eco, diet, lineage, tuple(mutations))

