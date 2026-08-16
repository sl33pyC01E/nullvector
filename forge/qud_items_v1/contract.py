from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math


FORMAT = "nullvector-qud-items-v1/1.0.0"
SLOTS = ("carapace", "manipulator", "sensor", "core")
MATERIALS = ("bone", "chitin", "living_wood", "iron", "glass", "crystal", "phase_fiber", "neural_gel")
COMPONENTS = (
    "edge", "reservoir", "lens", "actuator", "capacitor", "filter",
    "lattice", "spore_chamber", "memory_knot", "stabilizer", "grip", "shell",
)
EFFECTS = ("damage", "repair", "perception", "locomotion", "armor", "harvest", "phase", "fertility")


@dataclass(frozen=True, slots=True)
class Artifact:
    artifact_id: str
    name: str
    slot: str
    material: str
    components: tuple[str, ...]
    effects: tuple[tuple[str, float], ...]
    quality: float
    durability: float
    seed: int
    provenance: str

    def __post_init__(self) -> None:
        if not self.artifact_id or not self.name or self.slot not in SLOTS or self.material not in MATERIALS:
            raise ValueError("artifact identity drifted")
        if not 2 <= len(self.components) <= 6 or any(value not in COMPONENTS for value in self.components):
            raise ValueError("artifact components drifted")
        if len(set(name for name, _ in self.effects)) != len(self.effects):
            raise ValueError("artifact effect names are duplicated")
        if any(name not in EFFECTS or not math.isfinite(value) or value <= 0 for name, value in self.effects):
            raise ValueError("artifact effects drifted")
        if not 0 < self.quality <= 1 or not 0 <= self.durability <= 1 or not self.provenance:
            raise ValueError("artifact quality drifted")

    def effect(self, name: str) -> float:
        return next((value for key, value in self.effects if key == name), 0.0)

    def semantic_sha256(self) -> str:
        payload = {
            "format": FORMAT, "id": self.artifact_id, "name": self.name, "slot": self.slot,
            "material": self.material, "components": self.components, "effects": self.effects,
            "quality": self.quality, "durability": self.durability, "seed": self.seed,
            "provenance": self.provenance,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class Recipe:
    recipe_id: str
    name: str
    slot: str
    costs: tuple[tuple[str, float], ...]
    required_components: tuple[str, ...]
    effect_bias: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.recipe_id or not self.name or self.slot not in SLOTS or not self.costs:
            raise ValueError("recipe identity drifted")
        if any(not resource or not math.isfinite(amount) or amount <= 0 for resource, amount in self.costs):
            raise ValueError("recipe costs drifted")
        if any(component not in COMPONENTS for component in self.required_components):
            raise ValueError("recipe components drifted")
        if any(effect not in EFFECTS for effect in self.effect_bias):
            raise ValueError("recipe effect bias drifted")
