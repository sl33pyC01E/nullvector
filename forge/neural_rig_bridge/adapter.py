from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from ..morphology.constants import CANVAS_SIZE, SAFETY_MARGIN
from ..morphology.motion import FACING_DEGREES, FACING_NAMES
from .hashing import aligned_fields_hash, array_hash, canonical_json_hash
from .model import (
    ADAPTER_FORMAT,
    BACKGROUND_DRIVER,
    DRIVER_INDEX,
    DRIVER_NAMES,
    FRAME_FORMAT,
    BindingRejected,
    BoundRigFrame,
    NeuralRigBinding,
    readonly_array,
)


DEFAULT_Z_ORDER = (
    "appendage",
    "left_leg",
    "right_leg",
    "body",
    "left_arm",
    "right_arm",
    "head",
    "weapon",
)


def identity_matrix() -> np.ndarray:
    return np.eye(3, dtype=np.float64)


def _around(point: tuple[int, int], degrees: float) -> np.ndarray:
    x, y = map(float, point)
    radians = math.radians(degrees)
    cosine, sine = math.cos(radians), math.sin(radians)
    translate_to = np.asarray(
        ((1.0, 0.0, x), (0.0, 1.0, y), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    rotate = np.asarray(
        ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    translate_from = np.asarray(
        ((1.0, 0.0, -x), (0.0, 1.0, -y), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    return translate_to @ rotate @ translate_from


def _matrix(values: np.ndarray | Sequence[Sequence[float]], name: str) -> np.ndarray:
    try:
        raw = np.asarray(values)
    except (TypeError, ValueError) as error:
        raise BindingRejected(
            [f"transform {name} must be a rectangular numeric 3x3 matrix"]
        ) from error
    if raw.shape != (3, 3) or raw.dtype.kind not in "fiu":
        raise BindingRejected([f"transform {name} must be a numeric 3x3 matrix"])
    matrix = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise BindingRejected([f"transform {name} must be a finite 3x3 matrix"])
    if not np.allclose(matrix[2], (0.0, 0.0, 1.0), atol=1.0e-12, rtol=0.0):
        raise BindingRejected([f"transform {name} must be a 2D affine matrix"])
    matrix[2] = (0.0, 0.0, 1.0)
    matrix[np.abs(matrix) < 1.0e-15] = 0.0
    if bool((np.abs(matrix[:2, :2]) > 8.0).any()) or bool(
        (np.abs(matrix[:2, 2]) > CANVAS_SIZE * 4).any()
    ):
        raise BindingRejected([f"transform {name} exceeds the bounded affine domain"])
    determinant = float(np.linalg.det(matrix[:2, :2]))
    singular_values = np.linalg.svd(matrix[:2, :2], compute_uv=False)
    if (
        not math.isfinite(determinant)
        or abs(determinant) < 1.0e-9
        or not np.isfinite(singular_values).all()
        or float(singular_values.min()) < 0.25
        or float(singular_values.max()) > 4.0
        or float(singular_values.max() / singular_values.min()) > 16.0
    ):
        raise BindingRejected([f"transform {name} is singular"])
    return matrix


def _canonical_matrix(matrix: np.ndarray, name: str) -> np.ndarray:
    """Return the exact finite matrix persisted in a frame manifest.

    Rasterization must consume this value, never a higher-precision precursor,
    so exact manifest replay cannot cross a nearest-pixel half-step boundary.
    """
    canonical = np.round(np.asarray(matrix, dtype=np.float64), decimals=12)
    canonical[canonical == 0.0] = 0.0
    return _matrix(canonical, name)


def _transforms(
    transforms: Mapping[str, np.ndarray | Sequence[Sequence[float]]] | None,
) -> dict[str, np.ndarray]:
    if transforms is not None and not isinstance(transforms, Mapping):
        raise BindingRejected(["transforms must be a driver-to-matrix mapping"])
    supplied = dict(transforms or {})
    unknown = sorted(set(supplied) - set(DRIVER_NAMES))
    if unknown:
        raise BindingRejected([f"unknown motion drivers: {unknown}"])
    return {
        name: _matrix(supplied.get(name, identity_matrix()), name)
        for name in DRIVER_NAMES
    }


def _z_order(values: Sequence[str] | None) -> tuple[str, ...]:
    if values is not None and (
        isinstance(values, (str, bytes)) or not isinstance(values, Sequence)
    ):
        raise BindingRejected(["z_order must be a driver sequence"])
    result = tuple(DEFAULT_Z_ORDER if values is None else values)
    if len(result) != len(DRIVER_NAMES) or set(result) != set(DRIVER_NAMES):
        raise BindingRejected(["z_order must contain every driver exactly once"])
    return result


def _require_valid_binding(binding: NeuralRigBinding) -> None:
    # Local import avoids the binding -> validation -> adapter cycle while
    # keeping every public motion entry point fail-closed.
    from .validation import assert_valid_binding

    assert_valid_binding(binding)


def _tuples(values: tuple[np.ndarray, np.ndarray, np.ndarray]) -> set[tuple[int, int, int]]:
    return {
        tuple(map(int, row))
        for row in np.stack(values, axis=-1).reshape(-1, 3)
    }


def motion_adapter_contract(
    binding: NeuralRigBinding, *, facing: str = "north"
) -> dict[str, Any]:
    _require_valid_binding(binding)
    if facing not in FACING_NAMES:
        raise BindingRejected([f"unsupported facing {facing!r}"])
    root = binding.joints["root"].point
    base = {
        "format": ADAPTER_FORMAT,
        "binding_sha256": binding.sha256,
        "motion_renderer_version": "graph-layer-rig-v1",
        "facing": facing,
        "facing_degrees": float(FACING_DEGREES[facing]),
        "root": list(root),
        "driver_names": list(DRIVER_NAMES),
        "z_order": list(DEFAULT_Z_ORDER),
        "pivots": {
            node["id"]: list(node["pivot"])
            for node in binding.manifest["graph"]["nodes"]
        },
        "joints": {
            name: list(anchor.point) for name, anchor in binding.joints.items()
        },
        "sockets": {
            name: list(anchor.point) for name, anchor in binding.sockets.items()
        },
        "graph_edges": list(binding.manifest["graph"]["edges"]),
        "matrix_input": "source_to_destination_affine_3x3",
        "sampling": "inverse_nearest_neighbor",
        "tuple_policy": "copy_existing_tuple_only",
        "rest_frame_exact": True,
        "procedural_pixel_substitution": False,
    }
    base["adapter_sha256"] = canonical_json_hash(base)
    return base


def _facing_transforms_validated(
    binding: NeuralRigBinding, facing: str
) -> dict[str, np.ndarray]:
    if facing not in FACING_NAMES:
        raise BindingRejected([f"unsupported facing {facing!r}"])
    if facing == "north":
        return {name: identity_matrix() for name in DRIVER_NAMES}
    root = binding.joints["root"].point
    rotation = _around(root, FACING_DEGREES[facing])
    points = np.argwhere(binding.part_owner != 0)
    homogeneous = np.stack(
        (
            points[:, 1].astype(np.float64),
            points[:, 0].astype(np.float64),
            np.ones(len(points), dtype=np.float64),
        )
    )
    rotated = rotation @ homogeneous
    min_x, max_x = float(rotated[0].min()), float(rotated[0].max())
    min_y, max_y = float(rotated[1].min()), float(rotated[1].max())
    safe_min = float(SAFETY_MARGIN)
    safe_max = float(CANVAS_SIZE - SAFETY_MARGIN - 1)
    span = safe_max - safe_min
    scale = min(
        1.0,
        span / max(1.0, max_x - min_x),
        span / max(1.0, max_y - min_y),
    )
    scale_matrix = np.asarray(
        ((scale, 0.0, 0.0), (0.0, scale, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    scaled_min_x, scaled_max_x = scale * min_x, scale * max_x
    scaled_min_y, scaled_max_y = scale * min_y, scale * max_y
    desired_tx = float(root[0]) * (1.0 - scale)
    desired_ty = float(root[1]) * (1.0 - scale)
    tx = min(max(desired_tx, safe_min - scaled_min_x), safe_max - scaled_max_x)
    ty = min(max(desired_ty, safe_min - scaled_min_y), safe_max - scaled_max_y)
    translation = np.asarray(
        ((1.0, 0.0, tx), (0.0, 1.0, ty), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    transform = translation @ scale_matrix @ rotation
    return {name: transform.copy() for name in DRIVER_NAMES}


def facing_transforms(
    binding: NeuralRigBinding, facing: str
) -> dict[str, np.ndarray]:
    """Return one clip-wide fitted global facing transform for every driver."""
    _require_valid_binding(binding)
    return _facing_transforms_validated(binding, facing)


def _render_bound_pose_validated(
    binding: NeuralRigBinding,
    transforms: Mapping[str, np.ndarray | Sequence[Sequence[float]]] | None = None,
    *,
    z_order: Sequence[str] | None = None,
    enforce_margin: bool = True,
) -> BoundRigFrame:
    """Transform bound tuple layers without generating or relabelling a pixel."""
    matrices = {
        name: _canonical_matrix(matrix, name)
        for name, matrix in _transforms(transforms).items()
    }
    order = _z_order(z_order)
    part = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)
    material = np.zeros_like(part)
    emission = np.zeros_like(part)
    drivers = np.full_like(part, BACKGROUND_DRIVER)
    yy, xx = np.mgrid[0:CANVAS_SIZE, 0:CANVAS_SIZE]
    destinations = np.stack(
        (
            xx.reshape(-1).astype(np.float64),
            yy.reshape(-1).astype(np.float64),
            np.ones(CANVAS_SIZE * CANVAS_SIZE, dtype=np.float64),
        )
    )

    for driver_name in order:
        driver_id = DRIVER_INDEX[driver_name]
        inverse = np.linalg.inv(matrices[driver_name])
        source = inverse @ destinations
        source_x = np.rint(source[0]).astype(np.int64)
        source_y = np.rint(source[1]).astype(np.int64)
        inside = (
            (source_x >= 0)
            & (source_x < CANVAS_SIZE)
            & (source_y >= 0)
            & (source_y < CANVAS_SIZE)
        )
        destination_indices = np.flatnonzero(inside)
        sx = source_x[inside]
        sy = source_y[inside]
        owned = binding.driver_index[sy, sx] == driver_id
        destination_indices = destination_indices[owned]
        sx = sx[owned]
        sy = sy[owned]
        destination_y = destination_indices // CANVAS_SIZE
        destination_x = destination_indices % CANVAS_SIZE
        part[destination_y, destination_x] = binding.part_owner[sy, sx]
        material[destination_y, destination_x] = binding.material[sy, sx]
        emission[destination_y, destination_x] = binding.emission_level[sy, sx]
        drivers[destination_y, destination_x] = driver_id

    errors: list[str] = []
    if enforce_margin:
        visible = part != 0
        unsafe = bool(
            visible[:SAFETY_MARGIN].any()
            or visible[-SAFETY_MARGIN:].any()
            or visible[:, :SAFETY_MARGIN].any()
            or visible[:, -SAFETY_MARGIN:].any()
        )
        if unsafe:
            errors.append("transformed frame violates the safety margin")
    source_tuples = _tuples(
        (binding.part_owner, binding.material, binding.emission_level)
    )
    frame_tuples = _tuples((part, material, emission))
    introduced = sorted(frame_tuples - source_tuples)
    if introduced:
        errors.append(f"transform introduced tuples not present at rest: {introduced}")
    for driver_id, driver in enumerate(DRIVER_NAMES):
        if not bool((drivers == driver_id).any()):
            errors.append(f"transform erased driver {driver}")
    if errors:
        raise BindingRejected(errors)

    matrix_payload = {
        name: [[float(value) for value in row] for row in matrices[name]]
        for name in DRIVER_NAMES
    }
    base: dict[str, Any] = {
        "format": FRAME_FORMAT,
        "binding_sha256": binding.sha256,
        "raw_fields_sha256": aligned_fields_hash(part, material, emission),
        "driver_index_sha256": array_hash("posed_driver_index", drivers),
        "source_tuple_count": len(source_tuples),
        "frame_tuple_count": len(frame_tuples),
        "tuples_preserved": True,
        "procedural_pixel_substitution": False,
        "z_order": list(order),
        "source_to_destination": matrix_payload,
    }
    frame_hash_payload = {
        **base,
        "part_bytes_sha256": array_hash("posed_part", part),
        "material_bytes_sha256": array_hash("posed_material", material),
        "emission_bytes_sha256": array_hash("posed_emission", emission),
    }
    base["hashes"] = {"frame_sha256": canonical_json_hash(frame_hash_payload)}
    return BoundRigFrame(
        part_owner=readonly_array(part, dtype=np.uint8),
        material=readonly_array(material, dtype=np.uint8),
        emission_level=readonly_array(emission, dtype=np.uint8),
        driver_index=readonly_array(drivers, dtype=np.uint8),
        manifest=base,
    )


def render_bound_pose(
    binding: NeuralRigBinding,
    transforms: Mapping[str, np.ndarray | Sequence[Sequence[float]]] | None = None,
    *,
    z_order: Sequence[str] | None = None,
    enforce_margin: bool = True,
) -> BoundRigFrame:
    """Transform a completely validated binding without relabelling a tuple."""
    _require_valid_binding(binding)
    return _render_bound_pose_validated(
        binding,
        transforms,
        z_order=z_order,
        enforce_margin=enforce_margin,
    )


def validate_bound_frame(
    binding: NeuralRigBinding,
    frame: BoundRigFrame,
) -> list[str]:
    """Recompute a posed frame's complete categorical and manifest authority."""
    errors: list[str] = []
    try:
        _require_valid_binding(binding)
    except Exception as error:
        return [f"binding invalid: {error}"]
    if not isinstance(frame, BoundRigFrame):
        return ["frame must be a BoundRigFrame"]
    arrays = (
        ("part_owner", frame.part_owner),
        ("material", frame.material),
        ("emission_level", frame.emission_level),
        ("driver_index", frame.driver_index),
    )
    for name, values in arrays:
        if not isinstance(values, np.ndarray):
            errors.append(f"frame {name} must be a NumPy array")
        elif values.shape != (CANVAS_SIZE, CANVAS_SIZE) or values.dtype != np.uint8:
            errors.append(f"frame {name} must be uint8 [48, 48]")
    if errors:
        return errors
    part = np.asarray(frame.part_owner)
    material = np.asarray(frame.material)
    emission = np.asarray(frame.emission_level)
    drivers = np.asarray(frame.driver_index)
    if bool(((part == 0) != (drivers == BACKGROUND_DRIVER)).any()):
        errors.append("frame foreground and driver support disagree")
    if bool(((part == 0) & ((material != 0) | (emission != 0))).any()):
        errors.append("frame background carries material or emission")
    if bool((part > 16).any()) or bool((material > 9).any()) or bool((emission > 3).any()):
        errors.append("frame categorical value is out of vocabulary")
    if bool(((drivers != BACKGROUND_DRIVER) & (drivers >= len(DRIVER_NAMES))).any()):
        errors.append("frame driver index is out of vocabulary")
    for driver_id, driver in enumerate(DRIVER_NAMES):
        if not bool((drivers == driver_id).any()):
            errors.append(f"frame erased driver {driver}")
    source_tuples = _tuples(
        (binding.part_owner, binding.material, binding.emission_level)
    )
    frame_tuples = _tuples((part, material, emission))
    introduced = sorted(frame_tuples - source_tuples)
    if introduced:
        errors.append(f"frame introduced tuples not present at rest: {introduced}")
    visible = part != 0
    if bool(
        visible[:SAFETY_MARGIN].any()
        or visible[-SAFETY_MARGIN:].any()
        or visible[:, :SAFETY_MARGIN].any()
        or visible[:, -SAFETY_MARGIN:].any()
    ):
        errors.append("frame violates the safety margin")

    manifest = frame.manifest
    if not isinstance(manifest, Mapping):
        errors.append("frame manifest must be a mapping")
        return errors
    if manifest.get("format") != FRAME_FORMAT:
        errors.append("frame format is unsupported")
    if manifest.get("binding_sha256") != binding.sha256:
        errors.append("frame binding hash mismatch")
    expected_fields_hash = aligned_fields_hash(part, material, emission)
    if manifest.get("raw_fields_sha256") != expected_fields_hash:
        errors.append("frame raw-fields hash mismatch")
    expected_driver_hash = array_hash("posed_driver_index", drivers)
    if manifest.get("driver_index_sha256") != expected_driver_hash:
        errors.append("frame driver-index hash mismatch")
    if manifest.get("source_tuple_count") != len(source_tuples):
        errors.append("frame source tuple count mismatch")
    if manifest.get("frame_tuple_count") != len(frame_tuples):
        errors.append("frame tuple count mismatch")
    if manifest.get("tuples_preserved") is not True:
        errors.append("frame tuple-preservation flag is not true")
    if manifest.get("procedural_pixel_substitution") is not False:
        errors.append("frame procedural-substitution flag is not false")
    try:
        matrices = _transforms(manifest.get("source_to_destination"))
        order = _z_order(manifest.get("z_order"))
        expected = _render_bound_pose_validated(
            binding, matrices, z_order=order, enforce_margin=True
        )
    except Exception as error:
        errors.append(f"frame transform manifest is invalid: {error}")
        return errors
    if not all(
        np.array_equal(expected_array, actual_array)
        for expected_array, actual_array in (
            (expected.part_owner, part),
            (expected.material, material),
            (expected.emission_level, emission),
            (expected.driver_index, drivers),
        )
    ):
        errors.append("frame arrays are not the exact deterministic transform projection")
    if dict(expected.manifest) != dict(manifest):
        errors.append("frame manifest is not canonical")
    return errors


def assert_valid_bound_frame(
    binding: NeuralRigBinding,
    frame: BoundRigFrame,
) -> None:
    errors = validate_bound_frame(binding, frame)
    if errors:
        raise BindingRejected(errors)


def render_facing_frame(binding: NeuralRigBinding, facing: str) -> BoundRigFrame:
    _require_valid_binding(binding)
    return _render_bound_pose_validated(
        binding,
        _facing_transforms_validated(binding, facing),
    )
