from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Final

import numpy as np

from ..map_art.autotile import EAST, NORTH, SOUTH, WEST, cardinal_match_mask, elevation_drop_mask
from ..map_art.model import RENDERER_NAME, RENDERER_VERSION
from ..maps.model import HAZARD_NAMES, TERRAIN_NAMES, THEMES, MapData
from ..maps.validate import assert_valid
from .contract import CHANNEL_INDEX, FEATURE_CHANNELS, FEATURE_CONTRACT_SHA256, feature_manifest
from .hashing import array_sha256, coordinate_hash, named_arrays_sha256


_UINT64_MAX: Final[int] = (1 << 64) - 1
_NOISE_SALT: Final[int] = 0x4E4D4445434F5241
_CARDINAL_BITS: Final[tuple[int, ...]] = (NORTH, EAST, SOUTH, WEST)
_CARDINAL_OFFSETS: Final[tuple[tuple[int, int], ...]] = ((0, -1), (1, 0), (0, 1), (-1, 0))
_NOISE_SCALES: Final[tuple[int, ...]] = (1, 2, 4, 8)
_MAX_NAV_COST: Final[float] = 6.25


@dataclass(frozen=True, slots=True)
class FeatureInputs:
    protected_backbone: np.ndarray
    required_clearance: np.ndarray
    decoration_forbidden: np.ndarray


@dataclass(frozen=True, slots=True)
class EncodedFeatures:
    tensor: np.ndarray
    channel_manifest: dict[str, object]
    channel_manifest_sha256: str
    tensor_sha256: str
    input_mask_sha256: dict[str, str]
    input_masks_sha256: str
    public_seed: int
    map_id: str
    theme: str
    global_conditions: dict[str, object]

    def channel(self, name: str) -> np.ndarray:
        try:
            index = CHANNEL_INDEX[name]
        except KeyError as error:
            raise KeyError(f"Unknown feature channel {name!r}") from error
        view = self.tensor[index]
        view.setflags(write=False)
        return view

    def provenance(self) -> dict[str, object]:
        return {
            "map_id": self.map_id,
            "theme": self.theme,
            "public_seed": self.public_seed,
            "feature_contract_sha256": self.channel_manifest_sha256,
            "feature_tensor_sha256": self.tensor_sha256,
            "input_masks_sha256": self.input_masks_sha256,
            "input_mask_sha256": dict(self.input_mask_sha256),
            "global_conditions": dict(self.global_conditions),
        }


class FeatureValidationError(ValueError):
    pass


def _strict_mask(mask: np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
    if not isinstance(mask, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray; implicit conversion is forbidden.")
    if mask.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, observed {mask.shape}.")
    if mask.dtype != np.dtype(np.uint8):
        raise TypeError(f"{name} must use dtype uint8, observed {mask.dtype}.")
    if not bool(np.isin(mask, (0, 1)).all()):
        raise ValueError(f"{name} must be binary uint8.")
    return np.ascontiguousarray(mask)


def validate_feature_inputs(
    data: MapData,
    protected_backbone: np.ndarray,
    required_clearance: np.ndarray,
    decoration_forbidden: np.ndarray,
) -> FeatureInputs:
    """Validate masks without reconstructing or weakening their provenance."""
    assert_valid(data)
    if data.theme not in THEMES:
        raise ValueError(f"Unknown map theme {data.theme!r}.")
    shape = data.shape
    protected = _strict_mask(protected_backbone, shape, "protected_backbone")
    clearance = _strict_mask(required_clearance, shape, "required_clearance")
    forbidden = _strict_mask(decoration_forbidden, shape, "decoration_forbidden")
    required_union = (protected.astype(bool) | clearance.astype(bool) | (data.hazard != 0))
    if not bool(forbidden[required_union].all()):
        raise ValueError(
            "decoration_forbidden must include protected_backbone, required_clearance, and every hazard cell."
        )
    required_points = (data.start, data.exit, *data.objectives, *data.spawns)
    if not all(bool(clearance[y, x]) for x, y in required_points):
        raise ValueError("required_clearance must include start, exit, every objective, and every spawn.")
    return FeatureInputs(protected, clearance, forbidden)


def _direction_channels(bits: np.ndarray) -> tuple[np.ndarray, ...]:
    return tuple(((bits & bit) != 0).astype(np.float32) for bit in _CARDINAL_BITS)


def _zone_boundary_channels(zone: np.ndarray) -> tuple[np.ndarray, ...]:
    height, width = zone.shape
    channels: list[np.ndarray] = []
    for dx, dy in _CARDINAL_OFFSETS:
        boundary = np.ones((height, width), dtype=np.float32)
        y_dst = slice(max(0, -dy), min(height, height - dy))
        x_dst = slice(max(0, -dx), min(width, width - dx))
        y_src = slice(max(0, dy), min(height, height + dy))
        x_src = slice(max(0, dx), min(width, width + dx))
        boundary[y_dst, x_dst] = (zone[y_dst, x_dst] != zone[y_src, x_src]).astype(np.float32)
        channels.append(boundary)
    return tuple(channels)


def _point_mask(shape: tuple[int, int], points: tuple[tuple[int, int], ...]) -> np.ndarray:
    result = np.zeros(shape, dtype=np.float32)
    for x, y in points:
        result[y, x] = 1.0
    return result


def _distance_field(shape: tuple[int, int], points: tuple[tuple[int, int], ...]) -> np.ndarray:
    height, width = shape
    if not points:
        return np.ones(shape, dtype=np.float32)
    yy, xx = np.indices(shape, dtype=np.float64)
    squared = np.full(shape, np.inf, dtype=np.float64)
    for x, y in points:
        np.minimum(squared, (xx - x) ** 2 + (yy - y) ** 2, out=squared)
    diagonal = math.hypot(max(width - 1, 1), max(height - 1, 1))
    return np.ascontiguousarray(np.sqrt(squared) / diagonal, dtype=np.float32)


def _coordinates(shape: tuple[int, int]) -> tuple[np.ndarray, ...]:
    height, width = shape
    yy, xx = np.indices(shape, dtype=np.float64)
    x_norm = xx / max(width - 1, 1)
    y_norm = yy / max(height - 1, 1)
    center_x = (width - 1) * 0.5
    center_y = (height - 1) * 0.5
    max_radius = max(math.hypot(center_x, center_y), 1.0)
    radial = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2) / max_radius
    boundary_cells = np.minimum.reduce((xx, yy, width - 1 - xx, height - 1 - yy))
    boundary = boundary_cells / max((min(width, height) - 1) * 0.5, 1.0)
    return tuple(
        np.ascontiguousarray(np.clip(value, 0.0, 1.0), dtype=np.float32)
        for value in (x_norm, y_norm, radial, boundary)
    )


def _hash_unit(seed: int, x: int, y: int, salt: int) -> float:
    # Use the upper 24 bits: enough resolution for float32 and stable on every platform.
    return ((coordinate_hash(seed, x, y, salt) >> 40) / float((1 << 24) - 1)) * 2.0 - 1.0


def _noise_channel(shape: tuple[int, int], seed: int, scale: int, octave: int) -> np.ndarray:
    """Integer-bilinear coordinate-hash noise with no mutable RNG state."""
    height, width = shape
    result = np.empty(shape, dtype=np.float32)
    salt = _NOISE_SALT + scale * 0x101 + octave * 0x10001
    for y in range(height):
        y0 = (y // scale) * scale
        y1 = y0 + scale
        fy = (y - y0) / scale
        for x in range(width):
            x0 = (x // scale) * scale
            x1 = x0 + scale
            fx = (x - x0) / scale
            n00 = _hash_unit(seed, x0, y0, salt)
            n10 = _hash_unit(seed, x1, y0, salt)
            n01 = _hash_unit(seed, x0, y1, salt)
            n11 = _hash_unit(seed, x1, y1, salt)
            top = n00 + (n10 - n00) * fx
            bottom = n01 + (n11 - n01) * fx
            result[y, x] = top + (bottom - top) * fy
    return result


def encode_features(
    data: MapData,
    *,
    protected_backbone: np.ndarray,
    required_clearance: np.ndarray,
    decoration_forbidden: np.ndarray,
    public_seed: int,
) -> EncodedFeatures:
    inputs = validate_feature_inputs(
        data, protected_backbone, required_clearance, decoration_forbidden
    )
    if not isinstance(public_seed, (int, np.integer)):
        raise TypeError("public_seed must be an integer.")
    public_seed = int(public_seed)
    if not 0 <= public_seed <= _UINT64_MAX:
        raise ValueError("public_seed must be an unsigned 64-bit integer.")

    shape = data.shape
    channel_data: dict[str, np.ndarray] = {}
    for terrain_id in sorted(TERRAIN_NAMES):
        channel_data[f"terrain.{terrain_id}"] = (data.terrain == terrain_id).astype(np.float32)
    channel_data["walkability"] = data.walkability.astype(np.float32)
    for hazard_id in sorted(HAZARD_NAMES):
        channel_data[f"hazard.{hazard_id}"] = (data.hazard == hazard_id).astype(np.float32)
    channel_data["elevation.normalized"] = (
        data.elevation.astype(np.float32) - np.float32(2.5)
    ) / np.float32(2.5)

    for direction, array in zip(
        ("north", "east", "south", "west"),
        _direction_channels(elevation_drop_mask(data.elevation, data.walkability)),
        strict=True,
    ):
        channel_data[f"elevation_drop.{direction}"] = array
    for direction, array in zip(
        ("north", "east", "south", "west"),
        _direction_channels(cardinal_match_mask(data.terrain)),
        strict=True,
    ):
        channel_data[f"terrain_match.{direction}"] = array

    max_zone = int(data.zone.max(initial=-1))
    zone_normalized = np.zeros(shape, dtype=np.float32)
    if max_zone >= 0:
        valid_zone = data.zone >= 0
        zone_normalized[valid_zone] = (
            data.zone[valid_zone].astype(np.float32) + np.float32(1.0)
        ) / np.float32(max_zone + 1)
    channel_data["zone.normalized"] = zone_normalized
    for direction, array in zip(
        ("north", "east", "south", "west"), _zone_boundary_channels(data.zone), strict=True
    ):
        channel_data[f"zone_boundary.{direction}"] = array
    channel_data["nav_cost.log_normalized"] = np.ascontiguousarray(
        np.clip(np.log1p(data.nav_cost.astype(np.float64)) / math.log1p(_MAX_NAV_COST), 0.0, 1.0),
        dtype=np.float32,
    )
    channel_data["protected_backbone"] = inputs.protected_backbone.astype(np.float32)
    channel_data["required_clearance"] = inputs.required_clearance.astype(np.float32)
    channel_data["decoration_forbidden"] = inputs.decoration_forbidden.astype(np.float32)

    point_groups = {
        "start": (data.start,),
        "exit": (data.exit,),
        "objective": tuple(data.objectives),
        "spawn": tuple(data.spawns),
    }
    for name, points in point_groups.items():
        channel_data[f"required.{name}"] = _point_mask(shape, points)
    for name, points in point_groups.items():
        channel_data[f"distance.{name}"] = _distance_field(shape, points)

    for name, array in zip(
        ("coordinate.x", "coordinate.y", "coordinate.radial", "coordinate.boundary"),
        _coordinates(shape),
        strict=True,
    ):
        channel_data[name] = array
    for scale in _NOISE_SCALES:
        for octave_index, octave in enumerate(("a", "b")):
            channel_data[f"noise.scale_{scale}.{octave}"] = _noise_channel(
                shape, public_seed, scale, octave_index
            )

    ordered_names = tuple(channel.name for channel in FEATURE_CHANNELS)
    if set(channel_data) != set(ordered_names):
        missing = sorted(set(ordered_names) - set(channel_data))
        extra = sorted(set(channel_data) - set(ordered_names))
        raise RuntimeError(f"Feature channel construction drifted; missing={missing}, extra={extra}")
    tensor = np.ascontiguousarray(np.stack([channel_data[name] for name in ordered_names]), dtype=np.float32)
    if not bool(np.isfinite(tensor).all()):
        raise RuntimeError("Feature encoder produced a non-finite value.")
    manifest = feature_manifest()
    mask_arrays = {
        "protected_backbone": inputs.protected_backbone,
        "required_clearance": inputs.required_clearance,
        "decoration_forbidden": inputs.decoration_forbidden,
    }
    global_conditions = {
        "theme": data.theme,
        "theme_index": THEMES.index(data.theme),
        "map_width": data.config.width,
        "map_height": data.config.height,
        "aspect_ratio": data.config.width / data.config.height,
        "map_public_seed": int(data.seed),
        "feature_public_seed": public_seed,
        "feature_contract_version": str(manifest["contract_version"]),
        "feature_contract_sha256": FEATURE_CONTRACT_SHA256,
        "renderer_name": RENDERER_NAME,
        "renderer_version": RENDERER_VERSION,
    }
    # Local import avoids making catalog construction depend on feature encoding order.
    from .catalog import CATALOG_SHA256

    global_conditions["catalog_sha256"] = CATALOG_SHA256
    tensor.setflags(write=False)
    return EncodedFeatures(
        tensor=tensor,
        channel_manifest=manifest,
        channel_manifest_sha256=FEATURE_CONTRACT_SHA256,
        tensor_sha256=array_sha256(tensor),
        input_mask_sha256={name: array_sha256(array) for name, array in mask_arrays.items()},
        input_masks_sha256=named_arrays_sha256(mask_arrays),
        public_seed=public_seed,
        map_id=data.map_id,
        theme=data.theme,
        global_conditions=global_conditions,
    )


def validate_encoded_features(
    data: MapData,
    encoded: EncodedFeatures,
    *,
    protected_backbone: np.ndarray,
    required_clearance: np.ndarray,
    decoration_forbidden: np.ndarray,
) -> dict[str, object]:
    """Recompute the complete encoding and report any provenance or byte drift."""
    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    check("type", isinstance(encoded, EncodedFeatures), "encoded value uses EncodedFeatures")
    if not isinstance(encoded, EncodedFeatures):
        return {"passed": False, "checks": checks, "failures": ["type"]}
    expected = encode_features(
        data,
        protected_backbone=protected_backbone,
        required_clearance=required_clearance,
        decoration_forbidden=decoration_forbidden,
        public_seed=encoded.public_seed,
    )
    expected_shape = (len(FEATURE_CHANNELS), *data.shape)
    check("map_id", encoded.map_id == data.map_id, "map identity is bound to the source map")
    check("theme", encoded.theme == data.theme, "theme is bound to the source map")
    check("shape", encoded.tensor.shape == expected_shape, f"expected {expected_shape}")
    check("dtype", encoded.tensor.dtype == np.float32, "tensor dtype is float32")
    check("layout", encoded.tensor.flags.c_contiguous, "tensor is C-contiguous")
    check("finite", bool(np.isfinite(encoded.tensor).all()), "tensor contains finite values")
    check(
        "manifest",
        encoded.channel_manifest == feature_manifest(),
        "channel manifest exactly matches this encoder version",
    )
    check(
        "manifest_hash",
        encoded.channel_manifest_sha256 == FEATURE_CONTRACT_SHA256,
        "manifest hash matches this encoder version",
    )
    check(
        "tensor_hash",
        encoded.tensor_sha256 == array_sha256(encoded.tensor),
        "recorded tensor hash matches the observed tensor",
    )
    check(
        "tensor_exact",
        np.array_equal(encoded.tensor, expected.tensor),
        "tensor is byte-equivalent to a clean deterministic recomputation",
    )
    check(
        "mask_hashes",
        encoded.input_masks_sha256 == expected.input_masks_sha256
        and encoded.input_mask_sha256 == expected.input_mask_sha256,
        "all explicit topology mask hashes match",
    )
    check(
        "global_conditions",
        encoded.global_conditions == expected.global_conditions,
        "global conditions match the deterministic source contract",
    )
    if encoded.tensor.shape == expected_shape:
        for channel in FEATURE_CHANNELS:
            values = encoded.tensor[channel.index]
            check(
                f"range.{channel.name}",
                bool(((values >= channel.minimum) & (values <= channel.maximum)).all()),
                f"values are in [{channel.minimum}, {channel.maximum}]",
            )
    failures = [str(item["name"]) for item in checks if not item["passed"]]
    return {
        "passed": not failures,
        "map_id": data.map_id,
        "checks": checks,
        "failures": failures,
        "feature_contract_sha256": FEATURE_CONTRACT_SHA256,
        "feature_tensor_sha256": encoded.tensor_sha256,
    }


def assert_valid_encoded_features(
    data: MapData,
    encoded: EncodedFeatures,
    *,
    protected_backbone: np.ndarray,
    required_clearance: np.ndarray,
    decoration_forbidden: np.ndarray,
) -> dict[str, object]:
    report = validate_encoded_features(
        data,
        encoded,
        protected_backbone=protected_backbone,
        required_clearance=required_clearance,
        decoration_forbidden=decoration_forbidden,
    )
    if not report["passed"]:
        raise FeatureValidationError(
            "Encoded feature tensor failed validation: " + ", ".join(report["failures"])
        )
    return report
