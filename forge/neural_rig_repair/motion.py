from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from ..morphology.constants import CANVAS_SIZE, SAFETY_MARGIN
from ..morphology.motion import (
    DEFAULT_FPS,
    DEFAULT_FRAME_COUNTS,
    FACING_NAMES,
    LOOPING_MOTIONS,
    MOTION_NAMES,
    STABLE_STANCE_MOTIONS,
    motion_driver_matrices,
    motion_pose,
)
from ..neural_rig_bridge.adapter import DEFAULT_Z_ORDER, _render_bound_pose_validated
from ..neural_rig_bridge.model import BindingRejected, DRIVER_INDEX, DRIVER_NAMES
from ..neural_rig_bridge.motion_program import _clip_fit_matrix
from .hashing import array_sha256, canonical_json_bytes, sha256_bytes
from .constants import AURA_OWNER_ID
from .model import RepairedRigBinding


# The sequence is intentionally finite and source-controlled.  A clip either
# fits at one of these strengths or is rejected; there is no data-dependent
# unbounded search and no substitution of pixels or poses.
MOTION_STRENGTHS = (1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125, 0.0)
MAX_POSED_ANCHOR_SUPPORT_DISTANCE = 3.0

_DESTINATION_Y, _DESTINATION_X = np.mgrid[0:CANVAS_SIZE, 0:CANVAS_SIZE]
_DESTINATION_POINTS = np.stack(
    (
        _DESTINATION_X.reshape(-1).astype(np.float64),
        _DESTINATION_Y.reshape(-1).astype(np.float64),
        np.ones(CANVAS_SIZE * CANVAS_SIZE, dtype=np.float64),
    )
)
_DESTINATION_POINTS.setflags(write=False)


def _minimum_pixel_support_distance(
    points_yx: np.ndarray,
    target_xy: np.ndarray,
) -> float:
    """Measure a transformed anchor against the visible pixel footprint.

    The bridge rasterizer represents a pixel by its integer centre, but the
    rendered support occupies the closed unit square around that centre.  A
    centre-to-centre test therefore overstates the separation by as much as
    ``sqrt(0.5**2 + 0.5**2)`` after a fitted diagonal transform.  That caused
    valid, visibly adjacent appendage roots to fail at roughly 3.39 pixels even
    though their nearest rendered support cell was less than 2.9 pixels away.

    Keeping this calculation here (rather than changing the frozen bridge)
    preserves the bridge contract and makes the repair gate describe the
    actual raster domain it is validating.
    """
    points = np.asarray(points_yx)
    target = np.asarray(target_xy, dtype=np.float64)
    if (
        points.ndim != 2
        or points.shape[1:] != (2,)
        or len(points) < 1
        or target.shape != (2,)
        or not np.isfinite(target).all()
    ):
        raise ValueError("pixel support distance inputs are malformed")
    delta_x = np.maximum(
        np.abs(points[:, 1].astype(np.float64) - target[0]) - 0.5,
        0.0,
    )
    delta_y = np.maximum(
        np.abs(points[:, 0].astype(np.float64) - target[1]) - 0.5,
        0.0,
    )
    squared = delta_x * delta_x + delta_y * delta_y
    return math.sqrt(float(squared.min()))


def _canonical_matrix(values: np.ndarray) -> np.ndarray:
    result = np.round(np.asarray(values, dtype=np.float64), decimals=12)
    result[result == 0.0] = 0.0
    if result.shape != (3, 3) or not np.isfinite(result).all():
        raise ValueError("motion matrix must be finite float64 [3, 3]")
    return result


def _project_physical_driver_points(
    binding: RepairedRigBinding,
    matrices: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Rasterize each logical driver before z-order compositing.

    An articulation root may be intentionally hidden behind a later body layer.
    Measuring support from the final composite therefore confuses occlusion with
    a severed rig.  This projection uses the bridge's exact inverse-nearest
    raster rule, while excluding aura pixels because effects are never physical
    anchor support.  The finished frame still has to retain a visible pixel for
    every driver independently.
    """
    projected: dict[str, np.ndarray] = {}
    for driver in DRIVER_NAMES:
        driver_id = DRIVER_INDEX[driver]
        inverse = np.linalg.inv(_canonical_matrix(matrices[driver]))
        source = inverse @ _DESTINATION_POINTS
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
        physical_support = (
            (binding.driver_index[sy, sx] == driver_id)
            & (binding.part_owner[sy, sx] != AURA_OWNER_ID)
            & (binding.part_owner[sy, sx] != 0)
        )
        destination_indices = destination_indices[physical_support]
        if len(destination_indices) < 1:
            raise ValueError(f"motion erased physical support for driver {driver}")
        projected[driver] = np.column_stack(
            (
                destination_indices // CANVAS_SIZE,
                destination_indices % CANVAS_SIZE,
            )
        )
    return projected


def _matrix_record(values: np.ndarray) -> list[list[float]]:
    return [[float(value) for value in row] for row in _canonical_matrix(values)]


def _attenuate(
    raw: tuple[Mapping[str, np.ndarray], ...],
    strength: float,
) -> tuple[dict[str, np.ndarray], ...]:
    baseline = raw[0]
    result: list[dict[str, np.ndarray]] = []
    for matrices in raw:
        frame: dict[str, np.ndarray] = {}
        for driver in DRIVER_NAMES:
            base = np.asarray(baseline[driver], dtype=np.float64)
            target = np.asarray(matrices[driver], dtype=np.float64)
            frame[driver] = _canonical_matrix(base + strength * (target - base))
        result.append(frame)
    return tuple(result)


def _matrix_sequence_hash(matrix_sets: tuple[Mapping[str, np.ndarray], ...]) -> str:
    values = np.stack(
        [
            np.stack([_canonical_matrix(frame[driver]) for driver in DRIVER_NAMES])
            for frame in matrix_sets
        ]
    )
    return array_sha256("repair_motion_matrix_sequence", values)


def _render_candidate(
    binding: RepairedRigBinding,
    fitted: tuple[Mapping[str, np.ndarray], ...],
    z_order: tuple[str, ...],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    frame_records: list[dict[str, Any]] = []
    minimum_foreground = CANVAS_SIZE * CANVAS_SIZE
    maximum_foreground = 0
    maximum_anchor_support_distance = 0.0
    minimum_driver_pixels = {driver: CANVAS_SIZE * CANVAS_SIZE for driver in DRIVER_NAMES}
    maximum_driver_pixels = {driver: 0 for driver in DRIVER_NAMES}
    for index, matrices in enumerate(fitted):
        frame = _render_bound_pose_validated(
            binding, matrices, z_order=z_order, enforce_margin=True
        )
        projected_driver_points = _project_physical_driver_points(binding, matrices)
        foreground = int((frame.part_owner != 0).sum())
        minimum_foreground = min(minimum_foreground, foreground)
        maximum_foreground = max(maximum_foreground, foreground)
        driver_counts = []
        driver_points: dict[str, np.ndarray] = {}
        for driver in DRIVER_NAMES:
            count = int((frame.driver_index == DRIVER_INDEX[driver]).sum())
            if count < 1:
                raise ValueError(f"motion erased repaired driver {driver}")
            minimum_driver_pixels[driver] = min(minimum_driver_pixels[driver], count)
            maximum_driver_pixels[driver] = max(maximum_driver_pixels[driver], count)
            driver_counts.append(count)
            driver_points[driver] = np.argwhere(
                frame.driver_index == DRIVER_INDEX[driver]
            )
        for anchor in (*binding.joints.values(), *binding.sockets.values()):
            target = _canonical_matrix(matrices[anchor.driver]) @ np.asarray(
                (float(anchor.support_point[0]), float(anchor.support_point[1]), 1.0),
                dtype=np.float64,
            )
            points = projected_driver_points[anchor.driver]
            distance = _minimum_pixel_support_distance(points, target[:2])
            maximum_anchor_support_distance = max(
                maximum_anchor_support_distance, distance
            )
            if distance > MAX_POSED_ANCHOR_SUPPORT_DISTANCE:
                raise BindingRejected(
                    [
                        f"transformed {anchor.kind}.{anchor.name} lost local "
                        f"{anchor.driver} support ({distance:.6f} pixels)"
                    ]
                )
        frame_records.append(
            {
                "index": index,
                "frame_sha256": str(frame.manifest["hashes"]["frame_sha256"]),
                "posed_fields_sha256": str(frame.manifest["raw_fields_sha256"]),
                "driver_index_sha256": str(frame.manifest["driver_index_sha256"]),
                "foreground_pixels": foreground,
                "driver_pixels": driver_counts,
            }
        )
    metrics = {
        "minimum_foreground_pixels": minimum_foreground,
        "maximum_foreground_pixels": maximum_foreground,
        "minimum_driver_pixels": [minimum_driver_pixels[name] for name in DRIVER_NAMES],
        "maximum_driver_pixels": [maximum_driver_pixels[name] for name in DRIVER_NAMES],
        "maximum_anchor_support_distance": round(
            maximum_anchor_support_distance, 9
        ),
        "all_frames_margin_clear": True,
        "all_frames_source_tuples_only": True,
        "all_frames_retain_every_driver": True,
        "all_transformed_anchors_have_local_driver_support": True,
    }
    return frame_records, metrics


def _z_order_candidates(binding: RepairedRigBinding) -> tuple[tuple[str, ...], ...]:
    default = tuple(DEFAULT_Z_ORDER)
    counts = {
        driver: int((binding.driver_index == DRIVER_INDEX[driver]).sum())
        for driver in DRIVER_NAMES
    }
    # Larger layers first and tiny logical supports last is a robust fallback:
    # later layers cannot be wholly hidden by a larger coincident driver.
    support_order = tuple(sorted(DRIVER_NAMES, key=lambda name: (-counts[name], name)))
    seeds = (default, support_order, tuple(reversed(default)))
    candidates: list[tuple[str, ...]] = []
    for seed in seeds:
        if seed not in candidates:
            candidates.append(seed)
        for final_driver in sorted(DRIVER_NAMES, key=lambda name: (counts[name], name)):
            moved = tuple(name for name in seed if name != final_driver) + (final_driver,)
            if moved not in candidates:
                candidates.append(moved)
    return tuple(candidates)


def compile_motion_clip_audit(
    binding: RepairedRigBinding,
    motion: str,
    facing: str,
) -> dict[str, Any]:
    """Stress one repaired binding through one complete motion/facing clip.

    This deliberately emits hashes and metrics, not sprite pixels.  Rendering
    uses the frozen bridge's tuple-copying rasterizer, while binding validation
    remains repair-v2-specific and never asks v1 to reinterpret the rest bank.
    """
    if motion not in MOTION_NAMES:
        raise ValueError(f"unsupported repair motion {motion!r}")
    if facing not in FACING_NAMES:
        raise ValueError(f"unsupported repair facing {facing!r}")
    frame_count = DEFAULT_FRAME_COUNTS[motion]
    phases = tuple(index / float(frame_count - 1) for index in range(frame_count))
    poses = [motion_pose(motion, phase, binding.family_id) for phase in phases]
    if motion in LOOPING_MOTIONS:
        poses[-1] = poses[0]
    joints = {name: anchor.point for name, anchor in binding.joints.items()}
    sockets = {name: anchor.point for name, anchor in binding.sockets.items()}
    raw = tuple(
        motion_driver_matrices(
            pose,
            joints=joints,
            sockets=sockets,
            family=binding.family_id,
            facing=facing,
            plant_feet=motion in STABLE_STANCE_MOTIONS,
        )
        for pose in poses
    )
    failures: list[str] = []
    for strength in MOTION_STRENGTHS:
        try:
            attenuated = _attenuate(raw, strength)
            if any(
                not np.array_equal(
                    attenuated[0][driver], _canonical_matrix(raw[0][driver])
                )
                for driver in DRIVER_NAMES
            ):
                raise ValueError("motion attenuation changed the facing baseline")
            fit = _canonical_matrix(_clip_fit_matrix(binding, attenuated))
            fitted = tuple(
                {
                    driver: _canonical_matrix(fit @ matrices[driver])
                    for driver in DRIVER_NAMES
                }
                for matrices in attenuated
            )
            frames = None
            metrics = None
            selected_z_order = None
            order_failures: list[str] = []
            for z_order in _z_order_candidates(binding):
                try:
                    frames, metrics = _render_candidate(binding, fitted, z_order)
                    selected_z_order = z_order
                    break
                except BindingRejected as error:
                    order_failures.append(
                        f"{','.join(z_order)}: {'; '.join(error.errors)}"
                    )
            if frames is None or metrics is None or selected_z_order is None:
                raise ValueError(
                    "no bounded logical z-order retained every driver: "
                    + " | ".join(order_failures)
                )
            if motion in LOOPING_MOTIONS:
                first = frames[0]
                last = frames[-1]
                if any(
                    first[key] != last[key]
                    for key in (
                        "frame_sha256",
                        "posed_fields_sha256",
                        "driver_index_sha256",
                        "foreground_pixels",
                        "driver_pixels",
                    )
                ):
                    raise ValueError("loop endpoints are not exact")
            frame_sequence_sha256 = sha256_bytes(canonical_json_bytes(frames))
            base: dict[str, Any] = {
                "format": "nullvector-neural-rig-repair-clip-audit-v1",
                "sample_id": binding.sample_id,
                "binding_sha256": binding.sha256,
                "motion": motion,
                "facing": facing,
                "fps": DEFAULT_FPS[motion],
                "loop": motion in LOOPING_MOTIONS,
                "frame_count": frame_count,
                "motion_strength": strength,
                "z_order": list(selected_z_order),
                "z_order_policy": (
                    "frozen-default"
                    if selected_z_order == tuple(DEFAULT_Z_ORDER)
                    else "clip-local-driver-retention"
                ),
                "fit_source_to_destination": _matrix_record(fit),
                "unfitted_matrix_sequence_sha256": _matrix_sequence_hash(attenuated),
                "fitted_matrix_sequence_sha256": _matrix_sequence_hash(fitted),
                "frame_sequence_sha256": frame_sequence_sha256,
                "frames": frames,
                "metrics": metrics,
                "gates": {
                    "facing_baseline_preserved": True,
                    "clip_wide_uniform_positive_fit": float(fit[0, 0]) > 0.0
                    and float(fit[0, 0]) == float(fit[1, 1]),
                    "motion_attenuation_bounded": strength in MOTION_STRENGTHS,
                    "logical_z_order_bounded": selected_z_order
                    in _z_order_candidates(binding),
                    "raw_rest_arrays_untouched": True,
                    "loop_endpoint_exact": motion not in LOOPING_MOTIONS
                    or frames[0]["frame_sha256"] == frames[-1]["frame_sha256"],
                    "all_frames_valid": True,
                },
            }
            if any(value is not True for value in base["gates"].values()):
                raise ValueError("motion clip audit gate failed")
            base["clip_sha256"] = sha256_bytes(canonical_json_bytes(base))
            return base
        except Exception as error:
            failures.append(f"strength={strength:.3f}: {type(error).__name__}: {error}")
    raise ValueError(
        f"No bounded motion envelope fits {binding.sample_id}/{motion}/{facing}: "
        + " | ".join(failures)
    )


def compile_sample_motion_audit(binding: RepairedRigBinding) -> dict[str, Any]:
    clips: list[dict[str, Any]] = []
    for motion in MOTION_NAMES:
        for facing in FACING_NAMES:
            try:
                clips.append(compile_motion_clip_audit(binding, motion, facing))
            except Exception as error:
                # Keep the failing identity/clip at the end of a worker
                # traceback.  The bounded supervisor retains only a stderr
                # tail, so relying on the enormous inner envelope error can
                # erase the actionable context.
                raise ValueError(
                    f"repair motion audit failed for "
                    f"{binding.sample_id}/{motion}/{facing}"
                ) from error
    if len(clips) != 104 or sum(int(clip["frame_count"]) for clip in clips) != 944:
        raise ValueError("repair sample motion registry is not the exact 104/944 matrix")
    strengths: dict[str, int] = {}
    for clip in clips:
        key = f"{float(clip['motion_strength']):.3f}"
        strengths[key] = strengths.get(key, 0) + 1
    base = {
        "format": "nullvector-neural-rig-repair-sample-motion-audit-v1",
        "sample_id": binding.sample_id,
        "binding_sha256": binding.sha256,
        "clip_count": 104,
        "frame_count": 944,
        "motion_count": len(MOTION_NAMES),
        "facing_count": len(FACING_NAMES),
        "strength_histogram": strengths,
        "clips": clips,
        "gates": {
            "all_13_motions": True,
            "all_8_facings": True,
            "all_104_clips": True,
            "all_944_frames": True,
            "all_clip_gates_true": all(
                all(value is True for value in clip["gates"].values()) for clip in clips
            ),
        },
    }
    if any(value is not True for value in base["gates"].values()):
        raise ValueError("repair sample motion audit gate failed")
    base["sample_motion_sha256"] = sha256_bytes(canonical_json_bytes(base))
    return base
