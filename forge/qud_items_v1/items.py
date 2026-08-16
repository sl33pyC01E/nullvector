from __future__ import annotations

import hashlib
import numpy as np

from .contract import Artifact, COMPONENTS, EFFECTS, MATERIALS, Recipe, SLOTS


PREFIXES = ("Barbed", "Resonant", "Many-Jointed", "Pale", "Singing", "Hollow", "Verdant", "Phase-Wet", "Mnemonic", "Sun-Eating")
NOUNS = {
    "carapace": ("Shell", "Mantle", "Ward", "Cuirass"),
    "manipulator": ("Claw", "Cutter", "Hand", "Tendril"),
    "sensor": ("Eye", "Antenna", "Lens", "Crown"),
    "core": ("Heart", "Reactor", "Ganglion", "Seed"),
}
MATERIAL_EFFECT = {
    "bone": "armor", "chitin": "armor", "living_wood": "repair", "iron": "damage",
    "glass": "perception", "crystal": "phase", "phase_fiber": "locomotion", "neural_gel": "fertility",
}
COMPONENT_EFFECT = {
    "edge": "damage", "reservoir": "repair", "lens": "perception", "actuator": "locomotion",
    "capacitor": "damage", "filter": "repair", "lattice": "armor", "spore_chamber": "fertility",
    "memory_knot": "perception", "stabilizer": "phase", "grip": "harvest", "shell": "armor",
}

RECIPES = (
    Recipe("field_cutter", "Cellular Field Cutter", "manipulator", (("metal", 1.4), ("crystal", .35)), ("edge", "capacitor"), ("damage", "harvest")),
    Recipe("medulla_lens", "Medulla Survey Lens", "sensor", (("crystal", .8), ("biomass", .7)), ("lens", "memory_knot"), ("perception", "phase")),
    Recipe("living_shell", "Self-Knitting Shell", "carapace", (("biomass", 1.6), ("rock", .7)), ("shell", "reservoir"), ("armor", "repair")),
    Recipe("motive_core", "Ground-Coupled Motive Core", "core", (("metal", 1.2), ("water", .5)), ("actuator", "stabilizer"), ("locomotion", "repair")),
    Recipe("spore_heart", "Colonial Spore Heart", "core", (("biomass", 1.8), ("water", .9)), ("spore_chamber", "reservoir"), ("fertility", "repair")),
)


def generate_artifact(*, seed: int, provenance: str, slot: str | None = None, required_components: tuple[str, ...] = (), effect_bias: tuple[str, ...] = (), quality: float | None = None) -> Artifact:
    rng = np.random.default_rng(seed)
    chosen_slot = slot or SLOTS[int(rng.integers(len(SLOTS)))]
    material = MATERIALS[int(rng.integers(len(MATERIALS)))]
    count = int(rng.integers(2, 6))
    components = list(required_components)
    pool = [value for value in COMPONENTS if value not in components]
    rng.shuffle(pool)
    components.extend(pool[:max(0, count-len(components))])
    components = components[:6]
    q = float(np.clip(quality if quality is not None else rng.beta(3.2, 2.1), .08, 1))
    scores: dict[str, float] = {}
    scores[MATERIAL_EFFECT[material]] = .06 + q*.13
    for component in components:
        effect = COMPONENT_EFFECT[component]
        scores[effect] = scores.get(effect, 0.0) + (.025 + q*.052)
    for effect in effect_bias:
        scores[effect] = scores.get(effect, 0.0) + .045 + q*.04
    effects = tuple(sorted((name, round(value, 6)) for name, value in scores.items()))
    digest = hashlib.sha256(f"{seed}:{provenance}:{chosen_slot}:{material}:{components}".encode()).hexdigest()
    name = f"{PREFIXES[int(digest[:4], 16)%len(PREFIXES)]} {material.replace('_', ' ').title()} {NOUNS[chosen_slot][int(digest[4:8],16)%len(NOUNS[chosen_slot])]}"
    return Artifact(f"relic-{digest[:16]}", name, chosen_slot, material, tuple(components), effects, round(q, 6), 1.0, int(seed), provenance)


def craft(recipe: Recipe, *, seed: int, provenance: str, inventory: dict[str, float]) -> Artifact:
    missing = [name for name, amount in recipe.costs if inventory.get(name, 0.0) + 1e-9 < amount]
    if missing:
        raise ValueError("missing " + ", ".join(missing))
    for name, amount in recipe.costs:
        inventory[name] -= amount
    return generate_artifact(seed=seed, provenance=provenance, slot=recipe.slot, required_components=recipe.required_components, effect_bias=recipe.effect_bias, quality=.58)
