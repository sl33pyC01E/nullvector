from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntEnum, IntFlag
from typing import Final

from ..map_decorator.hashing import json_sha256


FORMAT: Final[str] = "nullvector-cellular-organism-bank-v1"
SPECIES_FORMAT: Final[str] = "nullvector-cellular-organism-species-v1"
ARRAY_FORMAT: Final[str] = "nullvector-cellular-organism-arrays-v1"
CANVAS_SIZE: Final[int] = 48
DISK_FLOOR_GIB: Final[float] = 100.0


class TissueType(IntEnum):
    EPIDERMIS = 1
    CONTRACTILE = 2
    STRUCTURAL = 3
    NEURAL = 4
    SENSORY = 5
    VASCULAR = 6
    DIGESTIVE = 7
    REPRODUCTIVE = 8
    STORAGE = 9
    ARMOR = 10
    WEAPON = 11
    EMITTER = 12
    IMMUNE = 13
    STEM = 14


class CellFlag(IntFlag):
    NONE = 0
    EYE = 1 << 0
    MOUTH = 1 << 1
    CIRCULATORY_CORE = 1 << 2
    REPRODUCTIVE = 1 << 3
    PHOTOSYNTHETIC = 1 << 4
    STEM = 1 << 5
    WEAPON = 1 << 6
    EMITTER = 1 << 7


TISSUE_NAMES: Final[tuple[str, ...]] = (
    "unused",
    "epidermis",
    "contractile",
    "structural",
    "neural",
    "sensory",
    "vascular",
    "digestive",
    "reproductive",
    "storage",
    "armor",
    "weapon",
    "emitter",
    "immune",
    "stem",
)

FLUID_BY_FAMILY: Final[tuple[str, ...]] = (
    "blood",
    "hemolymph",
    "sap",
    "phase_ichor",
    "coolant",
)


@dataclass(frozen=True, slots=True)
class SimulationDefaults:
    gravity: float = 28.0
    position_iterations: int = 5
    substeps: int = 2
    linear_damping: float = 0.985
    collision_restitution: float = 0.15
    fracture_impulse_scale: float = 0.018
    fluid_diffusion_rate: float = 2.2
    nutrient_diffusion_rate: float = 0.8
    leak_rate: float = 0.7
    starvation_damage_rate: float = 0.025
    regeneration_energy_cost: float = 0.45
    cell_pixel_scale: int = 5

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def contract_manifest() -> dict[str, object]:
    return {
        "format": FORMAT,
        "species_format": SPECIES_FORMAT,
        "array_format": ARRAY_FORMAT,
        "canvas_size": CANVAS_SIZE,
        "tissues": list(TISSUE_NAMES),
        "cell_flags": {name.lower(): int(value) for name, value in CellFlag.__members__.items()},
        "fluid_by_family": list(FLUID_BY_FAMILY),
        "arrays": {
            "position_xy": "int16[N,2]",
            "part_owner/material/emission/tissue": "uint8[N]",
            "organ_id": "uint16[N]",
            "cell_flags": "uint8[N]",
            "health/fluid/nutrient/energy/mass/stiffness": "float32[N]",
            "bond_ab": "uint16[M,2]",
            "bond_kind": "uint8[M]",
            "bond_rest/strength/conductance": "float32[M]",
        },
        "authority": {
            "one_physical_cell_per_non_aura_visible_source_pixel": True,
            "categorical_source_fields_immutable": True,
            "aura_owner_is_effect_not_physical_cell": True,
            "all_cells_have_tissue_and_organ": True,
            "all_graph_components_receive_explicit_bond_or_phase_tether": True,
        },
        "simulation_defaults": SimulationDefaults().to_dict(),
        "disk_floor_gib": DISK_FLOOR_GIB,
    }


CELLULAR_CONTRACT_SHA256: Final[str] = json_sha256(contract_manifest())
