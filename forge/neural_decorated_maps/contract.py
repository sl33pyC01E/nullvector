from __future__ import annotations

from typing import Final

from ..map_decorator.hashing import json_sha256
from ..map_decorator_production_v4_selection.contract import V4_SELECTION_CONTRACT_SHA256
from ..maps.model import THEMES


BANK_FORMAT: Final[str] = "nullvector-neural-decorated-map-bank/1.1.0"
RUNTIME_FORMAT: Final[str] = "nullvector-neural-decorated-map-runtime/1.1.0"
CELL_SIZE: Final[int] = 384
ATLAS_COLUMNS: Final[int] = 4
STATIC_LAYERS: Final[tuple[str, ...]] = (
    "composite",
    "base_color",
    "emissive",
    "objects",
    "variant",
    "emission_level",
    "topology",
)
ANIMATED_LAYER: Final[str] = "hazard"
HAZARD_FRAMES: Final[int] = 8


def contract_manifest() -> dict[str, object]:
    return {
        "format": "nullvector-neural-decorated-map-contract/1.1.0",
        "selection_contract_sha256": V4_SELECTION_CONTRACT_SHA256,
        "themes": list(THEMES),
        "atlas": {
            "cell_size": CELL_SIZE,
            "columns": ATLAS_COLUMNS,
            "static_layers": list(STATIC_LAYERS),
            "animated_layer": ANIMATED_LAYER,
            "hazard_frames": HAZARD_FRAMES,
            "filter": "nearest",
        },
        "authority": {
            "topology_v2_arrays_immutable": True,
            "source_points_immutable": True,
            "selection_checkpoint_not_shipped": True,
            "runtime_assets": [".json", ".png"],
            "python_runtime_required": False,
            "neural_heads_authorized": ["decal", "prop"],
            "semantic_heads_authorized": ["variant", "emission"],
            "unsupported_neural_heads_cross_runtime_boundary": False,
        },
        "rendering": {
            "tile_pixels": 8,
            "semantic_variant_drives_terrain_micro_pattern": True,
            "neural_object_classes_drive_catalog_sprites": True,
            "conditional_semantic_emission_scales_additive_pixels": True,
            "hazards_remain_topology_authoritative": True,
        },
    }


NEURAL_DECORATED_MAP_CONTRACT_SHA256: Final[str] = json_sha256(contract_manifest())
