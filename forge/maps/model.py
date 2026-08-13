from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import numpy as np


GENERATOR_NAME = "nullvector-map-forge"
GENERATOR_VERSION = "2.0.0"
MAP_SCHEMA_VERSION = "2.0.0"
RNG_NAME = "numpy.random.PCG64"
TOPOLOGY_MASK_CONTRACT_NAME = "nullvector-authoritative-map-topology-masks"
TOPOLOGY_MASK_CONTRACT_VERSION = "1.0.0"
TOPOLOGY_MASK_CAPTURE_POLICY = (
    "captured at generation-time mutation sites; never reconstructed"
)
TOPOLOGY_MASK_NAMES = (
    "protected_backbone",
    "required_clearance",
    "decoration_forbidden",
)
TOPOLOGY_MASK_MEANINGS = {
    "protected_backbone": (
        "Exact union of cells touched by the generation-time Manhattan-radius-two "
        "mission route carves from start to exit and every objective."
    ),
    "required_clearance": (
        "Exact union of generation-time safe regions cleared around start, exit, "
        "objectives, and spawn sockets."
    ),
    "decoration_forbidden": (
        "Exact union of protected_backbone, required_clearance, and final nonzero "
        "hazard cells."
    ),
}


class Terrain(IntEnum):
    VOID = 0
    FLOOR = 1
    WALL = 2
    WATER = 3
    BRIDGE = 4
    GROWTH = 5
    CRYSTAL = 6
    CHASM = 7
    SAND = 8


class Hazard(IntEnum):
    NONE = 0
    LASER = 1
    LAVA = 2
    SPORES = 3
    ARC = 4


THEMES = ("arena", "rooms", "caves", "archipelago", "garden", "anomaly")

WALKABLE_TERRAIN = frozenset(
    {
        int(Terrain.FLOOR),
        int(Terrain.BRIDGE),
        int(Terrain.GROWTH),
        int(Terrain.CRYSTAL),
        int(Terrain.SAND),
    }
)

TERRAIN_NAMES = {int(item): item.name.lower() for item in Terrain}
HAZARD_NAMES = {int(item): item.name.lower() for item in Hazard}


@dataclass(frozen=True, slots=True)
class MapConfig:
    width: int = 72
    height: int = 72
    objective_count: int = 3
    spawn_count: int = 12
    min_start_exit_distance: int = 0
    spawn_clearance_start: int = 8
    spawn_clearance_objective: int = 5
    spawn_clearance_hazard: int = 2

    def __post_init__(self) -> None:
        if not 32 <= self.width <= 256 or not 32 <= self.height <= 256:
            raise ValueError("Map dimensions must each be in [32, 256].")
        if not 1 <= self.objective_count <= 12:
            raise ValueError("objective_count must be in [1, 12].")
        if not 0 <= self.spawn_count <= 256:
            raise ValueError("spawn_count must be in [0, 256].")
        for label, value in (
            ("spawn_clearance_start", self.spawn_clearance_start),
            ("spawn_clearance_objective", self.spawn_clearance_objective),
            ("spawn_clearance_hazard", self.spawn_clearance_hazard),
        ):
            if value < 0:
                raise ValueError(f"{label} must be non-negative.")
        if self.min_start_exit_distance < 0:
            raise ValueError("min_start_exit_distance must be non-negative.")

    @property
    def effective_min_separation(self) -> int:
        if self.min_start_exit_distance:
            return self.min_start_exit_distance
        return max(12, min(self.width, self.height) // 2)

    def to_dict(self) -> dict[str, int]:
        return {
            "width": self.width,
            "height": self.height,
            "objective_count": self.objective_count,
            "spawn_count": self.spawn_count,
            "min_start_exit_distance": self.effective_min_separation,
            "spawn_clearance_start": self.spawn_clearance_start,
            "spawn_clearance_objective": self.spawn_clearance_objective,
            "spawn_clearance_hazard": self.spawn_clearance_hazard,
        }


Point = tuple[int, int]


@dataclass(slots=True)
class MapData:
    seed: int
    theme: str
    config: MapConfig
    terrain: np.ndarray
    walkability: np.ndarray
    hazard: np.ndarray
    elevation: np.ndarray
    zone: np.ndarray
    nav_cost: np.ndarray
    protected_backbone: np.ndarray
    required_clearance: np.ndarray
    decoration_forbidden: np.ndarray
    start: Point
    exit: Point
    objectives: tuple[Point, ...]
    spawns: tuple[Point, ...]
    repair_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return self.config.height, self.config.width

    @property
    def map_id(self) -> str:
        return f"{self.theme}-{self.seed:016x}-{self.config.width}x{self.config.height}"

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "terrain": np.ascontiguousarray(self.terrain, dtype=np.uint8),
            "walkability": np.ascontiguousarray(self.walkability, dtype=np.uint8),
            "hazard": np.ascontiguousarray(self.hazard, dtype=np.uint8),
            "elevation": np.ascontiguousarray(self.elevation, dtype=np.int8),
            "zone": np.ascontiguousarray(self.zone, dtype=np.int16),
            "nav_cost": np.ascontiguousarray(self.nav_cost, dtype=np.float32),
            "protected_backbone": np.ascontiguousarray(
                self.protected_backbone, dtype=np.uint8
            ),
            "required_clearance": np.ascontiguousarray(
                self.required_clearance, dtype=np.uint8
            ),
            "decoration_forbidden": np.ascontiguousarray(
                self.decoration_forbidden, dtype=np.uint8
            ),
        }


class MapInvariantError(ValueError):
    pass
