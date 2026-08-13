from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


RENDERER_NAME = "nullvector-map-art-forge"
RENDERER_VERSION = "1.0.0"
TILE_SIZE = 8
HAZARD_FRAME_COUNT = 8


RGB = tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class PropSpec:
    """A stable object-atlas entry and deterministic placement rule."""

    key: str
    shape: str
    kind: str
    allowed_terrain: tuple[int, ...]
    placement_modulus: int
    placement_slots: tuple[int, ...]
    collision: bool
    occlusion: int
    color_role: str = "primary"

    def __post_init__(self) -> None:
        if self.kind not in {"decal", "prop"}:
            raise ValueError("Prop kind must be 'decal' or 'prop'.")
        if self.placement_modulus < 1:
            raise ValueError("placement_modulus must be positive.")
        if not self.placement_slots:
            raise ValueError("At least one placement slot is required.")
        if any(slot < 0 or slot >= self.placement_modulus for slot in self.placement_slots):
            raise ValueError("Every placement slot must be inside placement_modulus.")
        if not 0 <= self.occlusion <= 3:
            raise ValueError("Occlusion class must be in [0, 3].")


@dataclass(frozen=True, slots=True)
class ThemeStyle:
    name: str
    terrain: tuple[RGB, ...]
    terrain_detail: tuple[RGB, ...]
    terrain_shadow: tuple[RGB, ...]
    edge_light: RGB
    edge_hot: RGB
    grid: RGB
    emission_primary: RGB
    emission_secondary: RGB
    hazard: tuple[RGB, ...]
    props: tuple[PropSpec, ...]

    def __post_init__(self) -> None:
        if len(self.terrain) != 9 or len(self.terrain_detail) != 9 or len(self.terrain_shadow) != 9:
            raise ValueError("Every theme must color all nine terrain semantics.")
        if len(self.hazard) != 5:
            raise ValueError("Hazard palette must contain NONE plus four hazard colors.")
        if len({spec.key for spec in self.props}) != len(self.props):
            raise ValueError(f"Theme {self.name!r} contains duplicate prop keys.")


@dataclass(frozen=True, slots=True)
class ArtInstance:
    instance_id: str
    catalog_index: int
    key: str
    kind: str
    cell: tuple[int, int]
    atlas_cell: tuple[int, int]
    collision: bool
    occlusion: int
    z_class: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "catalog_index": self.catalog_index,
            "key": self.key,
            "kind": self.kind,
            "cell": list(self.cell),
            "atlas_cell": list(self.atlas_cell),
            "collision": self.collision,
            "occlusion": self.occlusion,
            "z_class": self.z_class,
        }


@dataclass(slots=True)
class ArtLayers:
    """In-memory render product before atlas packing and publication."""

    base_color: np.ndarray
    emissive: np.ndarray
    hazard_color_frames: np.ndarray
    hazard_emissive_frames: np.ndarray
    autotile_mask: np.ndarray
    elevation_edge_mask: np.ndarray
    variant: np.ndarray
    collision: np.ndarray
    occlusion: np.ndarray
    prop_id: np.ndarray
    decal_id: np.ndarray
    instances: tuple[ArtInstance, ...]

    def semantic_arrays(self) -> dict[str, np.ndarray]:
        return {
            "autotile_mask": np.ascontiguousarray(self.autotile_mask, dtype=np.uint8),
            "elevation_edge_mask": np.ascontiguousarray(self.elevation_edge_mask, dtype=np.uint8),
            "variant": np.ascontiguousarray(self.variant, dtype=np.uint8),
            "collision": np.ascontiguousarray(self.collision, dtype=np.uint8),
            "occlusion": np.ascontiguousarray(self.occlusion, dtype=np.uint8),
            "prop_id": np.ascontiguousarray(self.prop_id, dtype=np.int16),
            "decal_id": np.ascontiguousarray(self.decal_id, dtype=np.int16),
        }

