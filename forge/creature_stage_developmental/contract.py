from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Final

from ..config import PROJECT_ROOT


FORMAT: Final[str] = "nullvector-creature-stage-developmental-organism-v1"
FAMILIES: Final[tuple[str, ...]] = ("humanoid", "animalian", "plantlike", "anomaly", "machine")
COMPONENT_KINDS: Final[tuple[str, ...]] = (
    "soma", "head", "pelvis", "sensor_crown", "mouth", "gut", "respirator",
    "circulator", "neural_cluster", "storage", "armor", "generator", "orbital",
)
APPENDAGE_KINDS: Final[tuple[str, ...]] = (
    "leg", "arm", "tail", "root", "frond", "tendril", "hardpoint", "wheel",
)
TRAITS: Final[tuple[str, ...]] = (
    "size", "symmetry", "segmentation", "stiffness", "elasticity", "bone_density",
    "muscle_density", "muscle_strength", "neural_density", "vascularity",
    "metabolism", "regeneration", "grip", "sensory_range", "phase_coherence",
)
TISSUES: Final[tuple[str, ...]] = (
    "skin", "bone", "muscle", "tendon", "armor", "neural", "vascular",
    "respiratory", "digestive", "sensor", "storage", "root", "phase", "machine",
    "weapon",
)
SCHEMA_PATH: Final[Path] = PROJECT_ROOT / "shared/schema/creature_stage_developmental_review.schema.json"
SOURCE_FILES: Final[tuple[str, ...]] = (
    "forge/creature_stage_developmental/contract.py",
    "forge/creature_stage_developmental/genomes.py",
    "forge/creature_stage_developmental/development.py",
    "forge/creature_stage_developmental/motion.py",
    "forge/creature_stage_developmental/review.py",
    "shared/schema/creature_stage_developmental_review.schema.json",
)


def _finite_tuple(values: tuple[float, ...], length: int, name: str) -> None:
    if len(values) != length or not all(math.isfinite(value) for value in values):
        raise ValueError(f"developmental {name} drifted")


@dataclass(frozen=True, slots=True)
class ComponentGene:
    component_id: str
    kind: str
    anchor: tuple[float, float]
    radius: tuple[float, float]
    parent: str | None = None
    side: int = 0
    trait_delta: tuple[float, ...] = (0.0,) * len(TRAITS)
    organ: str = "none"

    def __post_init__(self) -> None:
        if not self.component_id or self.kind not in COMPONENT_KINDS:
            raise ValueError("developmental component identity drifted")
        _finite_tuple(self.anchor, 2, "component anchor")
        _finite_tuple(self.radius, 2, "component radius")
        _finite_tuple(self.trait_delta, len(TRAITS), "component trait delta")
        if not all(0.6 <= value <= 12.0 for value in self.radius) or self.side not in (-1, 0, 1):
            raise ValueError("developmental component geometry drifted")


@dataclass(frozen=True, slots=True)
class AppendageGene:
    appendage_id: str
    kind: str
    root_component: str
    root_offset: tuple[float, float]
    endpoint: tuple[float, float]
    segments: int
    side: int
    phase: float
    bend: int = 1
    paired_with: str | None = None
    trait_delta: tuple[float, ...] = (0.0,) * len(TRAITS)

    def __post_init__(self) -> None:
        if not self.appendage_id or self.kind not in APPENDAGE_KINDS or not self.root_component:
            raise ValueError("developmental appendage identity drifted")
        _finite_tuple(self.root_offset, 2, "appendage root")
        _finite_tuple(self.endpoint, 2, "appendage endpoint")
        _finite_tuple(self.trait_delta, len(TRAITS), "appendage trait delta")
        if not 1 <= self.segments <= 5 or self.side not in (-1, 0, 1) or self.bend not in (-1, 1):
            raise ValueError("developmental appendage geometry drifted")
        if not 0.0 <= self.phase < 1.0:
            raise ValueError("developmental appendage phase drifted")


@dataclass(frozen=True, slots=True)
class DevelopmentalGenome:
    genome_id: str
    seed: int
    family_mix: tuple[float, ...]
    traits: tuple[float, ...]
    components: tuple[ComponentGene, ...]
    appendages: tuple[AppendageGene, ...]
    generation: int = 0
    parent_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.genome_id or type(self.seed) is not int or not 0 <= self.generation <= 1_000_000:
            raise ValueError("developmental genome identity drifted")
        _finite_tuple(self.family_mix, len(FAMILIES), "family mixture")
        _finite_tuple(self.traits, len(TRAITS), "global traits")
        if any(value < 0.0 for value in self.family_mix) or not math.isclose(sum(self.family_mix), 1.0, abs_tol=1e-6):
            raise ValueError("developmental family mixture must be a simplex")
        if any(not 0.0 <= value <= 1.0 for value in self.traits):
            raise ValueError("developmental global trait range drifted")
        if not 1 <= len(self.components) <= 32 or not 0 <= len(self.appendages) <= 32:
            raise ValueError("developmental component count drifted")
        component_ids = [component.component_id for component in self.components]
        appendage_ids = [appendage.appendage_id for appendage in self.appendages]
        if len(component_ids) != len(set(component_ids)) or len(appendage_ids) != len(set(appendage_ids)):
            raise ValueError("developmental component identifiers are not unique")
        known = set(component_ids)
        for component in self.components:
            if component.parent is not None and component.parent not in known:
                raise ValueError("developmental component parent is missing")
        for appendage in self.appendages:
            if appendage.root_component not in known:
                raise ValueError("developmental appendage root is missing")
            if appendage.paired_with is not None and appendage.paired_with not in set(appendage_ids):
                raise ValueError("developmental appendage pair is missing")
        appendage_lookup = {appendage.appendage_id: appendage for appendage in self.appendages}
        for appendage in self.appendages:
            if appendage.paired_with is None:
                continue
            partner = appendage_lookup[appendage.paired_with]
            if partner.paired_with != appendage.appendage_id:
                raise ValueError("developmental appendage pair is not reciprocal")
            if partner.kind != appendage.kind or partner.side != -appendage.side or partner.root_component != appendage.root_component:
                raise ValueError("developmental appendage pair is not anatomically symmetric")


def source_sha256() -> str:
    payload = {
        "format": FORMAT,
        "families": FAMILIES,
        "components": COMPONENT_KINDS,
        "appendages": APPENDAGE_KINDS,
        "traits": TRAITS,
        "tissues": TISSUES,
    }
    digest = hashlib.sha256(b"nullvector-developmental-organism-source-v1\0")
    digest.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii"))
    for relative in SOURCE_FILES:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        digest.update(relative.encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()
