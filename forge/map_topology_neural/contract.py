from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Final

import numpy as np

from ..maps.model import THEMES, WALKABLE_TERRAIN, MapConfig, Point
from .hashing import json_sha256, named_arrays_sha256


CONTRACT_NAME: Final[str] = "nullvector-neural-map-topology-tensor"
CONTRACT_VERSION: Final[str] = "1.0.0"
PATCH_SCALE: Final[int] = 4
FIELD_ORDER: Final[tuple[str, ...]] = ("terrain", "hazard", "elevation")
FIELD_CLASS_COUNTS: Final[dict[str, int]] = {
    "terrain": 9,
    "hazard": 5,
    "elevation": 6,
}
POINT_CHANNELS: Final[tuple[str, ...]] = ("start", "exit", "objective", "spawn")
GLOBAL_CONDITION_NAMES: Final[tuple[str, ...]] = (
    *(f"theme.{theme}" for theme in THEMES),
    "dimensions.width_32_256",
    "dimensions.height_32_256",
    "dimensions.aspect_width_fraction",
    "counts.objectives_1_12",
    "counts.spawns_0_256",
    "requested.hazard_budget_0_1",
    "requested.openness_0_1",
    "requested.route_complexity_0_1",
)


def contract_manifest() -> dict[str, object]:
    return {
        "contract_name": CONTRACT_NAME,
        "contract_version": CONTRACT_VERSION,
        "authority": {
            "sampled_fields": list(FIELD_ORDER),
            "immutable_conditioning": ["point_heatmaps", "global_conditions"],
            "derived_only": [
                "walkability",
                "nav_cost",
                "zone",
                "protected_backbone",
                "required_clearance",
                "decoration_forbidden",
            ],
            "raw_is_runtime_map": False,
        },
        "categorical": {
            "layout": "field_y_x_uint8",
            "field_order": list(FIELD_ORDER),
            "class_counts": FIELD_CLASS_COUNTS,
            "padding": {
                "scale": PATCH_SCALE,
                "sides": ["right", "bottom"],
                "terrain": 0,
                "hazard": 0,
                "elevation": 0,
                "valid_mask_required": True,
            },
        },
        "point_heatmaps": {
            "layout": "channel_y_x_float32",
            "channel_order": list(POINT_CHANNELS),
            "encoding": "exact_binary_impulses; objectives/spawns are union channels",
            "padding": 0.0,
        },
        "global_conditions": {
            "layout": "float32_vector",
            "names": list(GLOBAL_CONDITION_NAMES),
            "range": [0.0, 1.0],
        },
        "dimensions": {"minimum": 32, "maximum": 256, "rectangular": True},
        "crop": "remove exact recorded right/bottom padding; never resize",
    }


CONTRACT_SHA256: Final[str] = json_sha256(contract_manifest())


@dataclass(frozen=True, slots=True)
class TopologyConditions:
    hazard_budget: float
    openness: float
    route_complexity: float

    def __post_init__(self) -> None:
        for label, value in (
            ("hazard_budget", self.hazard_budget),
            ("openness", self.openness),
            ("route_complexity", self.route_complexity),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{label} must be finite and in [0, 1].")

    def to_dict(self) -> dict[str, float]:
        return {
            "hazard_budget": float(self.hazard_budget),
            "openness": float(self.openness),
            "route_complexity": float(self.route_complexity),
        }


@dataclass(frozen=True, slots=True)
class TopologyTensor:
    categorical: np.ndarray
    point_heatmaps: np.ndarray
    valid_mask: np.ndarray
    global_conditions: np.ndarray
    theme_index: int
    original_height: int
    original_width: int
    padded_height: int
    padded_width: int
    pad_bottom: int
    pad_right: int
    contract_sha256: str
    tensor_sha256: str

    @property
    def crop_slices(self) -> tuple[slice, slice]:
        return slice(0, self.original_height), slice(0, self.original_width)

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "categorical": self.categorical,
            "point_heatmaps": self.point_heatmaps,
            "valid_mask": self.valid_mask,
            "global_conditions": self.global_conditions,
            "theme_index": np.asarray(self.theme_index, dtype=np.int64),
        }


def _validate_points(
    config: MapConfig,
    start: Point,
    exit_point: Point,
    objectives: tuple[Point, ...],
    spawns: tuple[Point, ...],
) -> None:
    if len(objectives) != config.objective_count or len(spawns) != config.spawn_count:
        raise ValueError("Point counts disagree with MapConfig.")
    required = (start, exit_point, *objectives)
    if len(set(required)) != len(required) or len(set(spawns)) != len(spawns):
        raise ValueError("Required points and spawn points must be unique within their groups.")
    for point in (*required, *spawns):
        if (
            not isinstance(point, tuple)
            or len(point) != 2
            or isinstance(point[0], bool)
            or isinstance(point[1], bool)
            or not isinstance(point[0], int)
            or not isinstance(point[1], int)
        ):
            raise TypeError("Points must be exact integer (x, y) tuples.")
        x, y = point
        if not 0 <= x < config.width or not 0 <= y < config.height:
            raise ValueError(f"Point {point!r} lies outside the configured map.")


def _global_vector(
    theme: str,
    config: MapConfig,
    conditions: TopologyConditions,
) -> np.ndarray:
    if theme not in THEMES:
        raise ValueError(f"Unknown theme {theme!r}.")
    vector = np.zeros((len(GLOBAL_CONDITION_NAMES),), dtype=np.float32)
    vector[THEMES.index(theme)] = 1.0
    offset = len(THEMES)
    vector[offset + 0] = (config.width - 32) / 224.0
    vector[offset + 1] = (config.height - 32) / 224.0
    vector[offset + 2] = config.width / float(config.width + config.height)
    vector[offset + 3] = config.objective_count / 12.0
    vector[offset + 4] = config.spawn_count / 256.0
    vector[offset + 5] = conditions.hazard_budget
    vector[offset + 6] = conditions.openness
    vector[offset + 7] = conditions.route_complexity
    return vector


def encode_topology_tensor(
    *,
    terrain: np.ndarray,
    hazard: np.ndarray,
    elevation: np.ndarray,
    theme: str,
    config: MapConfig,
    start: Point,
    exit: Point,
    objectives: tuple[Point, ...],
    spawns: tuple[Point, ...],
    conditions: TopologyConditions,
) -> TopologyTensor:
    expected_shape = (config.height, config.width)
    fields = {"terrain": terrain, "hazard": hazard, "elevation": elevation}
    expected_dtypes = {
        "terrain": np.dtype(np.uint8),
        "hazard": np.dtype(np.uint8),
        "elevation": np.dtype(np.int8),
    }
    for name, array in fields.items():
        if not isinstance(array, np.ndarray) or array.shape != expected_shape:
            raise TypeError(f"{name} must be an ndarray with shape {expected_shape}.")
        if array.dtype != expected_dtypes[name]:
            raise TypeError(f"{name} must use dtype {expected_dtypes[name]}.")
        if not bool(((array >= 0) & (array < FIELD_CLASS_COUNTS[name])).all()):
            raise ValueError(f"{name} contains an out-of-contract categorical ID.")
    _validate_points(config, start, exit, objectives, spawns)
    padded_height = ((config.height + PATCH_SCALE - 1) // PATCH_SCALE) * PATCH_SCALE
    padded_width = ((config.width + PATCH_SCALE - 1) // PATCH_SCALE) * PATCH_SCALE
    categorical = np.zeros(
        (len(FIELD_ORDER), padded_height, padded_width), dtype=np.uint8
    )
    for index, name in enumerate(FIELD_ORDER):
        categorical[index, : config.height, : config.width] = fields[name].astype(
            np.uint8, copy=False
        )
    points = np.zeros(
        (len(POINT_CHANNELS), padded_height, padded_width), dtype=np.float32
    )
    points[0, start[1], start[0]] = 1.0
    points[1, exit[1], exit[0]] = 1.0
    for x, y in objectives:
        points[2, y, x] = 1.0
    for x, y in spawns:
        points[3, y, x] = 1.0
    valid = np.zeros((1, padded_height, padded_width), dtype=np.uint8)
    valid[0, : config.height, : config.width] = 1
    global_vector = _global_vector(theme, config, conditions)
    arrays = {
        "categorical": categorical,
        "point_heatmaps": points,
        "valid_mask": valid,
        "global_conditions": global_vector,
        "theme_index": np.asarray(THEMES.index(theme), dtype=np.int64),
    }
    digest = named_arrays_sha256(arrays)
    for array in arrays.values():
        array.setflags(write=False)
    return TopologyTensor(
        categorical=categorical,
        point_heatmaps=points,
        valid_mask=valid,
        global_conditions=global_vector,
        theme_index=THEMES.index(theme),
        original_height=config.height,
        original_width=config.width,
        padded_height=padded_height,
        padded_width=padded_width,
        pad_bottom=padded_height - config.height,
        pad_right=padded_width - config.width,
        contract_sha256=CONTRACT_SHA256,
        tensor_sha256=digest,
    )


def crop_categorical(tensor: TopologyTensor, categorical: np.ndarray) -> dict[str, np.ndarray]:
    expected = (len(FIELD_ORDER), tensor.padded_height, tensor.padded_width)
    if categorical.shape != expected:
        raise ValueError(f"Decoded categorical tensor must have shape {expected}.")
    ys, xs = tensor.crop_slices
    cropped = categorical[:, ys, xs]
    return {
        "terrain": np.ascontiguousarray(cropped[0], dtype=np.uint8),
        "hazard": np.ascontiguousarray(cropped[1], dtype=np.uint8),
        "elevation": np.ascontiguousarray(cropped[2], dtype=np.int8),
    }


def inferred_conditions(
    terrain: np.ndarray,
    hazard: np.ndarray,
    *,
    route_complexity: float = 0.5,
) -> TopologyConditions:
    walkable = np.isin(terrain, tuple(WALKABLE_TERRAIN))
    return TopologyConditions(
        hazard_budget=float(np.count_nonzero(hazard) / max(hazard.size, 1)),
        openness=float(np.count_nonzero(walkable) / max(walkable.size, 1)),
        route_complexity=float(route_complexity),
    )
