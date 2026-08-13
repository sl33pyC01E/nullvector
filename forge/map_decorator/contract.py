from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from .hashing import json_sha256


FEATURE_CONTRACT_NAME: Final[str] = "nullvector-topology-locked-map-features"
FEATURE_CONTRACT_VERSION: Final[str] = "1.0.0"
CATALOG_CONTRACT_NAME: Final[str] = "nullvector-topology-locked-decoration-catalog"
CATALOG_CONTRACT_VERSION: Final[str] = "1.0.0"
CARDINAL_DIRECTIONS: Final[tuple[str, ...]] = ("north", "east", "south", "west")


@dataclass(frozen=True, slots=True)
class FeatureChannel:
    index: int
    name: str
    group: str
    encoding: str
    minimum: float
    maximum: float

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "name": self.name,
            "group": self.group,
            "encoding": self.encoding,
            "minimum": self.minimum,
            "maximum": self.maximum,
        }


def _build_channels() -> tuple[FeatureChannel, ...]:
    specs: list[tuple[str, str, str, float, float]] = []
    specs.extend(
        (f"terrain.{terrain_id}", "terrain", "one_hot", 0.0, 1.0)
        for terrain_id in range(9)
    )
    specs.append(("walkability", "walkability", "binary", 0.0, 1.0))
    specs.extend(
        (f"hazard.{hazard_id}", "hazard", "one_hot", 0.0, 1.0)
        for hazard_id in range(5)
    )
    specs.append(
        ("elevation.normalized", "elevation", "value_minus_2_5_div_2_5", -1.0, 1.0)
    )
    specs.extend(
        (f"elevation_drop.{direction}", "elevation_edges", "exact_cardinal_flag", 0.0, 1.0)
        for direction in CARDINAL_DIRECTIONS
    )
    specs.extend(
        (f"terrain_match.{direction}", "terrain_adjacency", "exact_cardinal_flag", 0.0, 1.0)
        for direction in CARDINAL_DIRECTIONS
    )
    specs.append(("zone.normalized", "zone", "zone_plus_1_div_max_plus_1", 0.0, 1.0))
    specs.extend(
        (f"zone_boundary.{direction}", "zone_boundary", "exact_cardinal_flag", 0.0, 1.0)
        for direction in CARDINAL_DIRECTIONS
    )
    specs.append(("nav_cost.log_normalized", "navigation_cost", "log1p_div_log1p_6_25", 0.0, 1.0))
    specs.extend(
        (name, "protection", "binary", 0.0, 1.0)
        for name in ("protected_backbone", "required_clearance", "decoration_forbidden")
    )
    specs.extend(
        (f"required.{kind}", "required_points", "binary", 0.0, 1.0)
        for kind in ("start", "exit", "objective", "spawn")
    )
    specs.extend(
        (f"distance.{kind}", "distance_fields", "euclidean_div_map_diagonal", 0.0, 1.0)
        for kind in ("start", "exit", "objective", "spawn")
    )
    specs.extend(
        (
            name,
            "coordinates",
            encoding,
            0.0,
            1.0,
        )
        for name, encoding in (
            ("coordinate.x", "x_div_width_minus_1"),
            ("coordinate.y", "y_div_height_minus_1"),
            ("coordinate.radial", "distance_from_center_normalized"),
            ("coordinate.boundary", "distance_to_boundary_normalized"),
        )
    )
    specs.extend(
        (
            f"noise.scale_{scale}.{octave}",
            "seeded_noise",
            f"coordinate_hash_bilinear_lattice_scale_{scale}",
            -1.0,
            1.0,
        )
        for scale in (1, 2, 4, 8)
        for octave in ("a", "b")
    )
    channels = tuple(
        FeatureChannel(index, name, group, encoding, minimum, maximum)
        for index, (name, group, encoding, minimum, maximum) in enumerate(specs)
    )
    if len(channels) != 53:
        raise RuntimeError(f"Feature contract must contain 53 channels, observed {len(channels)}")
    return channels


FEATURE_CHANNELS: Final[tuple[FeatureChannel, ...]] = _build_channels()
CHANNEL_INDEX: Final[MappingProxyType[str, int]] = MappingProxyType(
    {channel.name: channel.index for channel in FEATURE_CHANNELS}
)


def feature_manifest() -> dict[str, object]:
    return {
        "contract_name": FEATURE_CONTRACT_NAME,
        "contract_version": FEATURE_CONTRACT_VERSION,
        "layout": "channels_first_float32_C_H_W",
        "cardinal_order": list(CARDINAL_DIRECTIONS),
        "channel_count": len(FEATURE_CHANNELS),
        "channels": [channel.to_dict() for channel in FEATURE_CHANNELS],
        "noise": {
            "hash": "splitmix64-coordinate-v1",
            "scales": [1, 2, 4, 8],
            "channels_per_scale": 2,
            "interpolation": "integer-bilinear",
        },
        "point_coordinates": "points are [x,y], tensor is [channel,y,x]",
    }


FEATURE_CONTRACT_SHA256: Final[str] = json_sha256(feature_manifest())
