from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from ..morphology.motion import (
    DEFAULT_FRAME_COUNTS,
    LOOPING_MOTIONS,
    STABLE_STANCE_MOTIONS,
    motion_driver_matrices,
    motion_event_specs,
    motion_pose,
)
from ..neural_rig_bridge.adapter import _render_bound_pose_validated
from ..neural_rig_bridge.hashing import canonical_json_hash
from ..neural_rig_bridge.model import DRIVER_INDEX, DRIVER_NAMES, BoundRigFrame
from ..neural_rig_bridge.motion_program import _transform_anchor
from ..neural_rig_repair.model import RepairedRigBinding
from ..neural_rig_repair.motion import _attenuate, _canonical_matrix, _matrix_sequence_hash


FRAME_FORMAT = "nullvector-neural-rig-repair-style-frame-v1"
CLIP_FORMAT = "nullvector-neural-rig-repair-style-clip-v1"


@dataclass(frozen=True, slots=True)
class RepairedMotionFrame:
    index: int
    phase: float
    pose: Any
    fields: BoundRigFrame
    joints: Mapping[str, tuple[int, int]]
    sockets: Mapping[str, tuple[int, int]]
    emission_pulse: int
    sha256: str


@dataclass(frozen=True, slots=True)
class RepairedMotionClip:
    binding: RepairedRigBinding
    motion: str
    facing: str
    fps: int
    loop: bool
    frames: tuple[RepairedMotionFrame, ...]
    manifest: Mapping[str, Any]

    @property
    def sha256(self) -> str:
        return str(self.manifest["clip_sha256"])


def _readonly_points(values: Mapping[str, tuple[int, int]]) -> Mapping[str, tuple[int, int]]:
    return MappingProxyType({name: tuple(map(int, point)) for name, point in values.items()})


def _anchor_points(
    anchors: Mapping[str, Any], matrices: Mapping[str, np.ndarray], fields: BoundRigFrame
) -> Mapping[str, tuple[int, int]]:
    return _readonly_points(
        {
            name: _transform_anchor(
                anchor.point,
                matrices[anchor.driver],
                fields.driver_index == DRIVER_INDEX[anchor.driver],
            )
            for name, anchor in anchors.items()
        }
    )


def _frame_record(
    index: int,
    phase: float,
    pose: Any,
    fields: BoundRigFrame,
    joints: Mapping[str, tuple[int, int]],
    sockets: Mapping[str, tuple[int, int]],
) -> dict[str, Any]:
    return {
        "format": FRAME_FORMAT,
        "index": int(index),
        "phase": round(float(phase), 9),
        "bound_frame_sha256": fields.sha256,
        "posed_fields_sha256": str(fields.manifest["raw_fields_sha256"]),
        "driver_index_sha256": str(fields.manifest["driver_index_sha256"]),
        "joints": {name: list(point) for name, point in sorted(joints.items())},
        "sockets": {name: list(point) for name, point in sorted(sockets.items())},
        "emission_pulse": int(pose.emission_pulse),
    }


def reconstruct_clip(
    binding: RepairedRigBinding,
    audit: Mapping[str, Any],
) -> RepairedMotionClip:
    """Reconstruct one styled-frame source from a sealed repair clip audit."""

    motion = str(audit["motion"])
    facing = str(audit["facing"])
    frame_count = DEFAULT_FRAME_COUNTS[motion]
    if (
        audit["format"] != "nullvector-neural-rig-repair-clip-audit-v1"
        or audit["sample_id"] != binding.sample_id
        or audit["binding_sha256"] != binding.sha256
        or audit["frame_count"] != frame_count
        or len(audit["frames"]) != frame_count
        or any(value is not True for value in audit["gates"].values())
    ):
        raise ValueError("repair style clip audit linkage failed")

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
    attenuated = _attenuate(raw, float(audit["motion_strength"]))
    fit = _canonical_matrix(np.asarray(audit["fit_source_to_destination"], dtype=np.float64))
    fitted = tuple(
        {
            driver: _canonical_matrix(fit @ matrices[driver])
            for driver in DRIVER_NAMES
        }
        for matrices in attenuated
    )
    if (
        _matrix_sequence_hash(attenuated) != audit["unfitted_matrix_sequence_sha256"]
        or _matrix_sequence_hash(fitted) != audit["fitted_matrix_sequence_sha256"]
    ):
        raise ValueError("repair style matrix replay differs from sealed motion authority")

    frames: list[RepairedMotionFrame] = []
    frame_records: list[dict[str, Any]] = []
    for index, (phase, pose, matrices, expected) in enumerate(
        zip(phases, poses, fitted, audit["frames"], strict=True)
    ):
        fields = _render_bound_pose_validated(
            binding,
            matrices,
            z_order=audit["z_order"],
            enforce_margin=True,
        )
        observed = {
            "index": index,
            "frame_sha256": fields.sha256,
            "posed_fields_sha256": str(fields.manifest["raw_fields_sha256"]),
            "driver_index_sha256": str(fields.manifest["driver_index_sha256"]),
            "foreground_pixels": int((fields.part_owner != 0).sum()),
            "driver_pixels": [
                int((fields.driver_index == DRIVER_INDEX[name]).sum())
                for name in DRIVER_NAMES
            ],
        }
        if observed != expected:
            raise ValueError(f"repair style frame {index} differs from sealed motion authority")
        frame_joints = _anchor_points(binding.joints, matrices, fields)
        frame_sockets = _anchor_points(binding.sockets, matrices, fields)
        record = _frame_record(index, phase, pose, fields, frame_joints, frame_sockets)
        frame_hash = canonical_json_hash(record)
        record["repair_style_frame_sha256"] = frame_hash
        frame_records.append(record)
        frames.append(
            RepairedMotionFrame(
                index=index,
                phase=phase,
                pose=pose,
                fields=fields,
                joints=frame_joints,
                sockets=frame_sockets,
                emission_pulse=int(pose.emission_pulse),
                sha256=frame_hash,
            )
        )
    if motion in LOOPING_MOTIONS and (
        frames[0].fields.sha256 != frames[-1].fields.sha256
        or frames[0].joints != frames[-1].joints
        or frames[0].sockets != frames[-1].sockets
        or frames[0].emission_pulse != frames[-1].emission_pulse
    ):
        raise ValueError("repair style loop endpoint is not exact")

    events = [
        {
            "name": name,
            "frame": min(frame_count - 1, int(round(float(phase) * (frame_count - 1)))),
            "phase": float(phase),
            "socket": socket,
        }
        for name, phase, socket in motion_event_specs(motion)
    ]
    manifest: dict[str, Any] = {
        "format": CLIP_FORMAT,
        "id": f"{binding.sample_id}__{motion}__{facing}",
        "sample_id": binding.sample_id,
        "binding_sha256": binding.sha256,
        "repair_clip_audit_sha256": audit["clip_sha256"],
        "motion": motion,
        "facing": facing,
        "fps": int(audit["fps"]),
        "loop": bool(audit["loop"]),
        "frame_count": frame_count,
        "motion_strength": float(audit["motion_strength"]),
        "z_order": list(audit["z_order"]),
        "fit_source_to_destination": audit["fit_source_to_destination"],
        "events": events,
        "frames": frame_records,
        "gates": {
            "sealed_audit_link_exact": True,
            "matrix_sequences_exact": True,
            "all_bound_frames_exact": True,
            "categorical_pixels_untouched": True,
            "loop_endpoint_exact": True,
        },
    }
    manifest["clip_sha256"] = canonical_json_hash(manifest)
    return RepairedMotionClip(
        binding=binding,
        motion=motion,
        facing=facing,
        fps=int(audit["fps"]),
        loop=bool(audit["loop"]),
        frames=tuple(frames),
        manifest=MappingProxyType(manifest),
    )
