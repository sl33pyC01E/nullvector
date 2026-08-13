from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from jsonschema import Draft202012Validator

from ..morphology.constants import CANVAS_SIZE, SAFETY_MARGIN
from ..morphology.motion import (
    DEFAULT_FPS,
    DEFAULT_FRAME_COUNTS,
    FACING_NAMES,
    LOOPING_MOTIONS,
    MOTION_NAMES,
    MOTION_RENDERER_VERSION,
    STABLE_STANCE_MOTIONS,
    MotionPose,
    motion_driver_matrices,
    motion_event_specs,
    motion_pose,
)
from .adapter import _render_bound_pose_validated, validate_bound_frame
from .hashing import binder_source_hash, canonical_json_hash
from .model import (
    DRIVER_INDEX,
    DRIVER_NAMES,
    BoundRigFrame,
    NeuralRigBinding,
)
from .validation import assert_valid_binding


NEURAL_MOTION_FORMAT = "nullvector-neural-rig-motion-clip-v1"
NEURAL_MOTION_FRAME_FORMAT = "nullvector-neural-rig-motion-frame-v1"
NEURAL_MOTION_REPLAY_FORMAT = "nullvector-neural-rig-motion-replay-v1"
NEURAL_MOTION_PROGRAM_VERSION = "graph-driver-motion-program-v1"
NEURAL_MOTION_SCHEMA = (
    Path(__file__).resolve().parents[2]
    / "shared"
    / "schema"
    / "neural_rig_motion.schema.json"
)


@lru_cache(maxsize=1)
def _motion_validator() -> Draft202012Validator:
    schema = json.loads(NEURAL_MOTION_SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


@dataclass(frozen=True, slots=True)
class NeuralMotionFrame:
    index: int
    phase: float
    pose: MotionPose
    fields: BoundRigFrame
    joints: Mapping[str, tuple[int, int]]
    sockets: Mapping[str, tuple[int, int]]
    emission_pulse: int
    sha256: str


@dataclass(frozen=True, slots=True)
class NeuralMotionClip:
    binding: NeuralRigBinding
    motion: str
    facing: str
    fps: int
    loop: bool
    frames: tuple[NeuralMotionFrame, ...]
    manifest: Mapping[str, Any]

    @property
    def sha256(self) -> str:
        return str(self.manifest["hashes"]["clip_sha256"])


def _translation_scale(scale: float, tx: float, ty: float) -> np.ndarray:
    return np.asarray(
        ((scale, 0.0, tx), (0.0, scale, ty), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )


def _driver_points(binding: NeuralRigBinding) -> dict[str, np.ndarray]:
    points: dict[str, np.ndarray] = {}
    for driver in DRIVER_NAMES:
        values = np.argwhere(binding.driver_index == DRIVER_INDEX[driver])
        if not len(values):
            raise ValueError(f"validated binding has no pixels for driver {driver}")
        points[driver] = np.stack(
            (
                values[:, 1].astype(np.float64),
                values[:, 0].astype(np.float64),
                np.ones(len(values), dtype=np.float64),
            )
        )
    return points


def _clip_fit_matrix(
    binding: NeuralRigBinding,
    matrix_sets: tuple[Mapping[str, np.ndarray], ...],
) -> np.ndarray:
    points = _driver_points(binding)
    minimum_x = float("inf")
    minimum_y = float("inf")
    maximum_x = float("-inf")
    maximum_y = float("-inf")
    for matrices in matrix_sets:
        for driver in DRIVER_NAMES:
            transformed = np.asarray(matrices[driver], dtype=np.float64) @ points[driver]
            minimum_x = min(minimum_x, float(transformed[0].min()))
            minimum_y = min(minimum_y, float(transformed[1].min()))
            maximum_x = max(maximum_x, float(transformed[0].max()))
            maximum_y = max(maximum_y, float(transformed[1].max()))

    # Reserve an additional raster pixel beyond the categorical safety margin.
    # This gives the presentation compiler room for a crisp outline and bloom.
    safe_min = float(SAFETY_MARGIN + 1)
    safe_max = float(CANVAS_SIZE - SAFETY_MARGIN - 2)
    safe_span = safe_max - safe_min
    span_x = max(1.0, maximum_x - minimum_x)
    span_y = max(1.0, maximum_y - minimum_y)
    scale = min(1.0, safe_span / span_x, safe_span / span_y)
    if not math.isfinite(scale) or scale < 0.25:
        raise ValueError("motion clip cannot be fitted inside the bounded affine domain")

    root_x, root_y = map(float, binding.joints["root"].point)
    desired_tx = root_x - scale * root_x
    desired_ty = root_y - scale * root_y
    minimum_tx = safe_min - scale * minimum_x
    maximum_tx = safe_max - scale * maximum_x
    minimum_ty = safe_min - scale * minimum_y
    maximum_ty = safe_max - scale * maximum_y
    tx = min(max(desired_tx, minimum_tx), maximum_tx)
    ty = min(max(desired_ty, minimum_ty), maximum_ty)
    return _translation_scale(scale, tx, ty)


def _transform_anchor(
    point: tuple[int, int], matrix: np.ndarray, driver_mask: np.ndarray
) -> tuple[int, int]:
    target = np.asarray((float(point[0]), float(point[1]), 1.0))
    transformed = np.asarray(matrix, dtype=np.float64) @ target
    candidates = np.argwhere(driver_mask)
    if not len(candidates):
        raise ValueError("motion frame erased an anchor driver")
    distances = (
        (candidates[:, 1].astype(np.float64) - transformed[0]) ** 2
        + (candidates[:, 0].astype(np.float64) - transformed[1]) ** 2
    )
    y, x = candidates[int(np.argmin(distances))]
    return int(x), int(y)


def _readonly_points(
    points: Mapping[str, tuple[int, int]],
) -> Mapping[str, tuple[int, int]]:
    return MappingProxyType({name: tuple(map(int, point)) for name, point in points.items()})


def _frame_base(
    *,
    index: int,
    phase: float,
    fields: BoundRigFrame,
    joints: Mapping[str, tuple[int, int]],
    sockets: Mapping[str, tuple[int, int]],
    emission_pulse: int,
) -> dict[str, Any]:
    return {
        "format": NEURAL_MOTION_FRAME_FORMAT,
        "index": int(index),
        "phase": round(float(phase), 9),
        "bound_frame_sha256": fields.sha256,
        "raw_fields_sha256": str(fields.manifest["raw_fields_sha256"]),
        "driver_index_sha256": str(fields.manifest["driver_index_sha256"]),
        "joints": {name: list(point) for name, point in sorted(joints.items())},
        "sockets": {name: list(point) for name, point in sorted(sockets.items())},
        "emission_pulse": int(emission_pulse),
    }


def _make_frame(
    *,
    index: int,
    phase: float,
    pose: MotionPose,
    fields: BoundRigFrame,
    binding: NeuralRigBinding,
    matrices: Mapping[str, np.ndarray],
) -> NeuralMotionFrame:
    joints = {
        name: _transform_anchor(
            anchor.point,
            matrices[anchor.driver],
            fields.driver_index == DRIVER_INDEX[anchor.driver],
        )
        for name, anchor in binding.joints.items()
    }
    sockets = {
        name: _transform_anchor(
            anchor.point,
            matrices[anchor.driver],
            fields.driver_index == DRIVER_INDEX[anchor.driver],
        )
        for name, anchor in binding.sockets.items()
    }
    readonly_joints = _readonly_points(joints)
    readonly_sockets = _readonly_points(sockets)
    base = _frame_base(
        index=index,
        phase=phase,
        fields=fields,
        joints=readonly_joints,
        sockets=readonly_sockets,
        emission_pulse=pose.emission_pulse,
    )
    return NeuralMotionFrame(
        index=index,
        phase=float(phase),
        pose=pose,
        fields=fields,
        joints=readonly_joints,
        sockets=readonly_sockets,
        emission_pulse=int(pose.emission_pulse),
        sha256=canonical_json_hash(base),
    )


def _replace_sockets(
    frame: NeuralMotionFrame, sockets: Mapping[str, tuple[int, int]]
) -> NeuralMotionFrame:
    readonly_sockets = _readonly_points(sockets)
    base = _frame_base(
        index=frame.index,
        phase=frame.phase,
        fields=frame.fields,
        joints=frame.joints,
        sockets=readonly_sockets,
        emission_pulse=frame.emission_pulse,
    )
    return replace(
        frame,
        sockets=readonly_sockets,
        sha256=canonical_json_hash(base),
    )


def _stabilize_planted_feet(
    frames: tuple[NeuralMotionFrame, ...],
) -> tuple[NeuralMotionFrame, ...]:
    stable: dict[str, tuple[int, int]] = {}
    for socket_name, driver in (
        ("left_foot", "left_leg"),
        ("right_foot", "right_leg"),
    ):
        driver_id = DRIVER_INDEX[driver]
        common = np.logical_and.reduce(
            [frame.fields.driver_index == driver_id for frame in frames]
        )
        if not bool(common.any()):
            raise ValueError(f"no common raster support for planted {socket_name}")
        mean_x = float(np.mean([frame.sockets[socket_name][0] for frame in frames]))
        mean_y = float(np.mean([frame.sockets[socket_name][1] for frame in frames]))
        points = np.argwhere(common)
        distances = (
            (points[:, 1].astype(np.float64) - mean_x) ** 2
            + (points[:, 0].astype(np.float64) - mean_y) ** 2
        )
        y, x = points[int(np.argmin(distances))]
        stable[socket_name] = (int(x), int(y))

    result: list[NeuralMotionFrame] = []
    for frame in frames:
        sockets = dict(frame.sockets)
        sockets.update(stable)
        result.append(_replace_sockets(frame, sockets))
    return tuple(result)


def _span(points: list[tuple[int, int]]) -> list[int]:
    return [
        max(point[0] for point in points) - min(point[0] for point in points),
        max(point[1] for point in points) - min(point[1] for point in points),
    ]


def _clip_manifest(
    binding: NeuralRigBinding,
    motion: str,
    facing: str,
    fps: int,
    frames: tuple[NeuralMotionFrame, ...],
    fit: np.ndarray,
) -> dict[str, Any]:
    events = [
        {
            "name": name,
            "frame": min(
                len(frames) - 1,
                int(round(float(phase) * float(len(frames) - 1))),
            ),
            "phase": float(phase),
            "socket": socket,
        }
        for name, phase, socket in motion_event_specs(motion)
    ]
    frame_records = []
    for frame in frames:
        record = _frame_base(
            index=frame.index,
            phase=frame.phase,
            fields=frame.fields,
            joints=frame.joints,
            sockets=frame.sockets,
            emission_pulse=frame.emission_pulse,
        )
        record["motion_frame_sha256"] = frame.sha256
        frame_records.append(record)
    base: dict[str, Any] = {
        "format": NEURAL_MOTION_FORMAT,
        "id": f"{binding.sample_id}__{motion}__{facing}",
        "motion_program_version": NEURAL_MOTION_PROGRAM_VERSION,
        "motion_renderer_version": MOTION_RENDERER_VERSION,
        "binder_source_sha256": binder_source_hash(),
        "binding_sha256": binding.sha256,
        "source_raw_fields_sha256": binding.raw_fields_sha256,
        "condition": {
            "family": binding.family,
            "family_id": binding.family_id,
            "subtype_id": binding.subtype_id,
            "role_id": binding.role_id,
        },
        "motion": motion,
        "facing": facing,
        "loop": motion in LOOPING_MOTIONS,
        "fps": int(fps),
        "frame_count": len(frames),
        "canvas_size": [CANVAS_SIZE, CANVAS_SIZE],
        "safety_margin": SAFETY_MARGIN,
        "driver_names": list(DRIVER_NAMES),
        "clip_fit_source_to_destination": [
            [round(float(value), 12) for value in row] for row in fit
        ],
        "events": events,
        "frames": frame_records,
        "metrics": {
            "unique_field_frames": len(
                {str(frame.fields.manifest["raw_fields_sha256"]) for frame in frames}
            ),
            "root_span": _span([frame.joints["root"] for frame in frames]),
            "left_foot_span": _span(
                [frame.sockets["left_foot"] for frame in frames]
            ),
            "right_foot_span": _span(
                [frame.sockets["right_foot"] for frame in frames]
            ),
            "all_source_tuples_preserved": True,
            "procedural_pixel_substitution": False,
            "margin_clear": True,
        },
    }
    base["hashes"] = {"clip_sha256": canonical_json_hash(base)}
    return base


def compile_neural_motion_clip(
    binding: NeuralRigBinding,
    motion: str,
    *,
    facing: str = "north",
    frame_count: int | None = None,
    fps: int | None = None,
) -> NeuralMotionClip:
    """Animate immutable neural fields with the shared graph motion program."""
    assert_valid_binding(binding)
    if motion not in MOTION_NAMES:
        raise ValueError(f"Unsupported motion {motion!r}; expected one of {MOTION_NAMES}")
    if facing not in FACING_NAMES:
        raise ValueError(f"Unsupported facing {facing!r}; expected one of {FACING_NAMES}")
    resolved_count = DEFAULT_FRAME_COUNTS[motion] if frame_count is None else frame_count
    if (
        isinstance(resolved_count, bool)
        or not isinstance(resolved_count, int)
        or not 3 <= resolved_count <= 256
    ):
        raise ValueError("frame_count must be an integer in [3, 256]")
    resolved_fps = DEFAULT_FPS[motion] if fps is None else fps
    if (
        isinstance(resolved_fps, bool)
        or not isinstance(resolved_fps, int)
        or not 1 <= resolved_fps <= 60
    ):
        raise ValueError("fps must be an integer in [1, 60]")

    phases = tuple(index / float(resolved_count - 1) for index in range(resolved_count))
    poses = [motion_pose(motion, phase, binding.family_id) for phase in phases]
    if motion in LOOPING_MOTIONS:
        poses[-1] = poses[0]
    joints = {name: anchor.point for name, anchor in binding.joints.items()}
    sockets = {name: anchor.point for name, anchor in binding.sockets.items()}
    raw_matrices = tuple(
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
    fit = _clip_fit_matrix(binding, raw_matrices)
    fitted_matrices = tuple(
        {driver: fit @ matrices[driver] for driver in DRIVER_NAMES}
        for matrices in raw_matrices
    )
    frames = tuple(
        _make_frame(
            index=index,
            phase=phases[index],
            pose=poses[index],
            fields=_render_bound_pose_validated(
                binding, fitted_matrices[index], enforce_margin=True
            ),
            binding=binding,
            matrices=fitted_matrices[index],
        )
        for index in range(resolved_count)
    )
    if motion in STABLE_STANCE_MOTIONS:
        frames = _stabilize_planted_feet(frames)
    manifest = _clip_manifest(
        binding, motion, facing, resolved_fps, frames, fit
    )
    clip = NeuralMotionClip(
        binding=binding,
        motion=motion,
        facing=facing,
        fps=resolved_fps,
        loop=motion in LOOPING_MOTIONS,
        frames=frames,
        manifest=MappingProxyType(manifest),
    )
    errors = validate_neural_motion_clip(clip)
    if errors:
        raise ValueError("; ".join(errors))
    return clip


def validate_neural_motion_clip(clip: NeuralMotionClip) -> list[str]:
    errors: list[str] = []
    if not isinstance(clip, NeuralMotionClip):
        return ["clip must be a NeuralMotionClip"]
    try:
        assert_valid_binding(clip.binding)
    except Exception as error:  # validation must report, not crash callers
        errors.append(f"binding invalid: {error}")
        return errors
    if clip.motion not in MOTION_NAMES:
        errors.append("motion is unsupported")
    if clip.facing not in FACING_NAMES:
        errors.append("facing is unsupported")
    if not clip.frames:
        errors.append("clip has no frames")
        return errors
    if clip.loop != (clip.motion in LOOPING_MOTIONS):
        errors.append("loop flag disagrees with motion program")
    source_tuples = set(
        map(
            tuple,
            np.stack(
                (
                    clip.binding.part_owner,
                    clip.binding.material,
                    clip.binding.emission_level,
                ),
                axis=-1,
            ).reshape(-1, 3),
        )
    )
    for expected_index, frame in enumerate(clip.frames):
        if frame.index != expected_index:
            errors.append(f"frame {expected_index} index mismatch")
        if frame.fields.manifest.get("binding_sha256") != clip.binding.sha256:
            errors.append(f"frame {expected_index} binding hash mismatch")
        for error in validate_bound_frame(clip.binding, frame.fields):
            errors.append(f"frame {expected_index} bound authority: {error}")
        frame_tuples = set(
            map(
                tuple,
                np.stack(
                    (
                        frame.fields.part_owner,
                        frame.fields.material,
                        frame.fields.emission_level,
                    ),
                    axis=-1,
                ).reshape(-1, 3),
            )
        )
        if not frame_tuples <= source_tuples:
            errors.append(f"frame {expected_index} introduced a tuple")
        for name, anchor in clip.binding.joints.items():
            x, y = frame.joints.get(name, (-1, -1))
            if not (0 <= x < CANVAS_SIZE and 0 <= y < CANVAS_SIZE) or int(
                frame.fields.driver_index[y, x]
            ) != DRIVER_INDEX[anchor.driver]:
                errors.append(f"frame {expected_index} joint {name} lacks driver support")
        for name, anchor in clip.binding.sockets.items():
            x, y = frame.sockets.get(name, (-1, -1))
            if not (0 <= x < CANVAS_SIZE and 0 <= y < CANVAS_SIZE) or int(
                frame.fields.driver_index[y, x]
            ) != DRIVER_INDEX[anchor.driver]:
                errors.append(f"frame {expected_index} socket {name} lacks driver support")
        expected_base = _frame_base(
            index=frame.index,
            phase=frame.phase,
            fields=frame.fields,
            joints=frame.joints,
            sockets=frame.sockets,
            emission_pulse=frame.emission_pulse,
        )
        if canonical_json_hash(expected_base) != frame.sha256:
            errors.append(f"frame {expected_index} hash mismatch")
    if clip.loop:
        first, last = clip.frames[0], clip.frames[-1]
        if not all(
            np.array_equal(a, b)
            for a, b in (
                (first.fields.part_owner, last.fields.part_owner),
                (first.fields.material, last.fields.material),
                (first.fields.emission_level, last.fields.emission_level),
                (first.fields.driver_index, last.fields.driver_index),
            )
        ) or first.joints != last.joints or first.sockets != last.sockets:
            errors.append("loop endpoints are not exact")
    fit = np.asarray(clip.manifest.get("clip_fit_source_to_destination"), dtype=np.float64)
    if fit.shape != (3, 3):
        errors.append("clip fit matrix is invalid")
    else:
        expected_manifest = _clip_manifest(
            clip.binding, clip.motion, clip.facing, clip.fps, clip.frames, fit
        )
        if dict(clip.manifest) != expected_manifest:
            errors.append("clip manifest is not canonical")
    for error in sorted(
        _motion_validator().iter_errors(dict(clip.manifest)),
        key=lambda value: tuple(str(part) for part in value.absolute_path),
    ):
        path = ".".join(str(part) for part in error.absolute_path) or "$"
        errors.append(f"motion manifest schema {path}: {error.message}")
    return errors


def assert_valid_neural_motion_clip(clip: NeuralMotionClip) -> None:
    errors = validate_neural_motion_clip(clip)
    if errors:
        raise ValueError("Invalid neural motion clip: " + "; ".join(errors))


def replay_neural_motion_clip(clip: NeuralMotionClip) -> dict[str, Any]:
    """Recompile a clip and compare every categorical field and manifest byte."""
    errors: list[str] = []
    try:
        assert_valid_neural_motion_clip(clip)
        replayed = compile_neural_motion_clip(
            clip.binding,
            clip.motion,
            facing=clip.facing,
            frame_count=len(clip.frames),
            fps=clip.fps,
        )
    except Exception as error:
        return {
            "format": NEURAL_MOTION_REPLAY_FORMAT,
            "passed": False,
            "errors": [str(error)],
            "checks": {},
        }
    field_checks = []
    anchor_checks = []
    frame_hash_checks = []
    for expected, actual in zip(clip.frames, replayed.frames, strict=True):
        field_checks.append(
            all(
                np.array_equal(a, b)
                for a, b in (
                    (expected.fields.part_owner, actual.fields.part_owner),
                    (expected.fields.material, actual.fields.material),
                    (expected.fields.emission_level, actual.fields.emission_level),
                    (expected.fields.driver_index, actual.fields.driver_index),
                )
            )
        )
        anchor_checks.append(
            expected.joints == actual.joints and expected.sockets == actual.sockets
        )
        frame_hash_checks.append(expected.sha256 == actual.sha256)
    checks = {
        "frame_count_exact": len(clip.frames) == len(replayed.frames),
        "fields_exact": bool(all(field_checks)),
        "anchors_exact": bool(all(anchor_checks)),
        "frame_hashes_exact": bool(all(frame_hash_checks)),
        "manifest_exact": dict(clip.manifest) == dict(replayed.manifest),
        "clip_hash_exact": clip.sha256 == replayed.sha256,
    }
    if not all(checks.values()):
        errors.append("one or more replay checks failed")
    report: dict[str, Any] = {
        "format": NEURAL_MOTION_REPLAY_FORMAT,
        "passed": not errors,
        "errors": errors,
        "checks": checks,
        "binding_sha256": clip.binding.sha256,
        "clip_sha256": clip.sha256,
        "frame_count": len(clip.frames),
    }
    report["report_sha256"] = canonical_json_hash(report)
    return report


def assert_exact_neural_motion_replay(report: Mapping[str, Any]) -> None:
    expected_checks = {
        "frame_count_exact",
        "fields_exact",
        "anchors_exact",
        "frame_hashes_exact",
        "manifest_exact",
        "clip_hash_exact",
    }
    if (
        not isinstance(report, Mapping)
        or report.get("format") != NEURAL_MOTION_REPLAY_FORMAT
        or report.get("passed") is not True
        or report.get("errors") != []
        or set(report.get("checks", {})) != expected_checks
        or not all(value is True for value in report["checks"].values())
    ):
        raise ValueError("neural motion replay report is not exact")
    payload = dict(report)
    claimed = payload.pop("report_sha256", None)
    if claimed != canonical_json_hash(payload):
        raise ValueError("neural motion replay report hash mismatch")
