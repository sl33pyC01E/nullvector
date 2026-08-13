from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw

from .constants import (
    APPENDAGE,
    ARMOR,
    BODY,
    CANVAS_SIZE,
    CORE,
    DETAIL,
    EMISSION,
    EMISSION_LEVEL_NAMES,
    HEAD,
    JOINT_LAYER,
    LAYER_NAMES,
    LEFT_ARM,
    LEFT_LEG,
    MATERIAL_NAMES,
    PART_OWNER_NAMES,
    RIGHT_ARM,
    RIGHT_LEG,
    SAFETY_MARGIN,
    SOCKET_LAYER,
    STRUCTURAL_LAYERS,
    WEAPON,
)
from .contract import component_count, validate_specimen
from .fields import MorphologyTrainingFields, build_training_fields
from .render import MorphologySpecimen, compose_rgba, layers_to_tokens


MOTION_FORMAT = "neural-morphology-motion-manifest-v1"
MOTION_RENDERER_VERSION = "graph-layer-rig-v1"

MOTION_NAMES = (
    "idle_breathe",
    "idle_wiggle",
    "locomote",
    "joy",
    "anger",
    "fear",
    "confused",
    "sleep",
    "taunt",
    "attack",
    "cast",
    "hit",
    "death",
)

LOOPING_MOTIONS = frozenset(MOTION_NAMES[:9])
ACTION_MOTIONS = frozenset(MOTION_NAMES[9:])
STABLE_STANCE_MOTIONS = frozenset(("idle_breathe", "idle_wiggle", "sleep"))

FACING_NAMES = (
    "north",
    "northeast",
    "east",
    "southeast",
    "south",
    "southwest",
    "west",
    "northwest",
)

FACING_DEGREES = {
    name: float(index * 45) for index, name in enumerate(FACING_NAMES)
}

DEFAULT_FRAME_COUNTS = {
    "idle_breathe": 9,
    "idle_wiggle": 9,
    "locomote": 9,
    "joy": 9,
    "anger": 9,
    "fear": 13,
    "confused": 9,
    "sleep": 9,
    "taunt": 9,
    "attack": 8,
    "cast": 8,
    "hit": 7,
    "death": 10,
}

DEFAULT_FPS = {
    "idle_breathe": 8,
    "idle_wiggle": 10,
    "locomote": 12,
    "joy": 10,
    "anger": 12,
    "fear": 14,
    "confused": 9,
    "sleep": 6,
    "taunt": 11,
    "attack": 14,
    "cast": 12,
    "hit": 16,
    "death": 10,
}

_EVENT_SPECS: dict[str, tuple[tuple[str, float, str | None], ...]] = {
    "idle_breathe": (("inhale_peak", 0.25, "focus"), ("exhale_peak", 0.75, "focus")),
    "idle_wiggle": (("sway_left", 0.25, "appendage_tip"), ("sway_right", 0.75, "appendage_tip")),
    "locomote": (("left_plant", 0.0, "left_foot"), ("right_plant", 0.5, "right_foot")),
    "joy": (("emote_peak", 0.5, "focus"),),
    "anger": (("emote_peak", 0.5, "focus"),),
    "fear": (("emote_peak", 0.5, "focus"),),
    "confused": (("emote_peak", 0.5, "focus"),),
    "sleep": (("breath_peak", 0.25, "focus"),),
    "taunt": (("gesture_peak", 0.25, "right_hand"), ("gesture_return", 0.75, "right_hand")),
    "attack": (("windup", 0.14, "muzzle"), ("strike", 0.62, "muzzle"), ("recover", 1.0, "muzzle")),
    "cast": (("charge", 0.28, "focus"), ("release", 0.78, "muzzle")),
    "hit": (("impact", 0.33, "focus"), ("recover", 1.0, "focus")),
    "death": (("collapse", 0.22, "focus"), ("grounded", 1.0, "focus")),
}

_DRIVER_BY_LAYER = {
    BODY: "body",
    ARMOR: "body",
    HEAD: "head",
    LEFT_ARM: "left_arm",
    RIGHT_ARM: "right_arm",
    LEFT_LEG: "left_leg",
    RIGHT_LEG: "right_leg",
    APPENDAGE: "appendage",
    WEAPON: "weapon",
    CORE: "body",
}

_OWNER_PRIORITY = (
    BODY,
    ARMOR,
    HEAD,
    LEFT_ARM,
    RIGHT_ARM,
    LEFT_LEG,
    RIGHT_LEG,
    APPENDAGE,
    WEAPON,
    CORE,
)

_POINT_DRIVER = {
    "root": "body",
    "head": "head",
    "left_shoulder": "left_arm",
    "right_shoulder": "right_arm",
    "left_hip": "left_leg",
    "right_hip": "right_leg",
    "appendage_base": "appendage",
    "weapon_mount": "weapon",
    "focus": "body",
    "muzzle": "weapon",
    "left_hand": "left_arm",
    "right_hand": "right_arm",
    "left_foot": "left_leg",
    "right_foot": "right_leg",
    "appendage_tip": "appendage",
}


@dataclass(frozen=True, slots=True)
class MotionPose:
    root_dx: float = 0.0
    root_dy: float = 0.0
    global_angle: float = 0.0
    global_sx: float = 1.0
    global_sy: float = 1.0
    body_angle: float = 0.0
    body_sx: float = 1.0
    body_sy: float = 1.0
    head_angle: float = 0.0
    head_sx: float = 1.0
    head_sy: float = 1.0
    left_arm_angle: float = 0.0
    right_arm_angle: float = 0.0
    left_leg_angle: float = 0.0
    right_leg_angle: float = 0.0
    appendage_angle: float = 0.0
    weapon_angle: float = 0.0
    emission_pulse: int = 0


@dataclass(frozen=True, slots=True)
class MotionFrame:
    index: int
    phase: float
    layers: np.ndarray
    tokens: np.ndarray
    rgba: np.ndarray
    joints: dict[str, list[int]]
    sockets: dict[str, list[int]]
    sha256: str

    def training_fields(self, specimen: MorphologySpecimen) -> MorphologyTrainingFields:
        return build_training_fields(
            self.layers, specimen.genome, self.joints, self.sockets
        )


@dataclass(frozen=True, slots=True)
class MotionClip:
    specimen: MorphologySpecimen
    motion: str
    facing: str
    fps: int
    loop: bool
    frames: tuple[MotionFrame, ...]
    manifest: dict[str, Any]

    @property
    def sha256(self) -> str:
        return str(self.manifest["hashes"]["clip_sha256"])


@dataclass(frozen=True, slots=True)
class MotionBlend:
    first: str
    second: str
    weight: float
    phase: float
    pose: MotionPose


def blend_motion_poses(
    first: str,
    second: str,
    *,
    weight: float,
    phase: float,
    family: int,
) -> MotionBlend:
    """Deterministically interpolate two motion states before rasterization.

    This is the runtime-facing primitive for transitions such as locomote to
    attack or idle_breathe with a small fear overlay. It intentionally blends
    rig parameters, never finished pixels, preserving crisp categorical masks.
    """
    if first not in MOTION_NAMES or second not in MOTION_NAMES:
        raise ValueError("Both blended motions must belong to MOTION_NAMES")
    if not 0.0 <= weight <= 1.0 or not math.isfinite(weight):
        raise ValueError("weight must be finite and in [0, 1]")
    if not 0.0 <= phase <= 1.0 or not math.isfinite(phase):
        raise ValueError("phase must be finite and in [0, 1]")
    if isinstance(family, bool) or not isinstance(family, int) or not 0 <= family < 5:
        raise ValueError("family must be an integer in [0, 4]")
    first_pose = _pose_for(first, phase, family)
    second_pose = _pose_for(second, phase, family)
    values = {
        field: (1.0 - weight) * float(getattr(first_pose, field))
        + weight * float(getattr(second_pose, field))
        for field in MotionPose.__dataclass_fields__
        if field != "emission_pulse"
    }
    values["emission_pulse"] = (
        first_pose.emission_pulse if weight < 0.5 else second_pose.emission_pulse
    )
    return MotionBlend(
        first=first,
        second=second,
        weight=weight,
        phase=phase,
        pose=MotionPose(**values),
    )


def _translation(dx: float, dy: float) -> np.ndarray:
    return np.asarray(
        ((1.0, 0.0, dx), (0.0, 1.0, dy), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )


def _rotation(degrees: float) -> np.ndarray:
    radians = math.radians(degrees)
    cosine = math.cos(radians)
    sine = math.sin(radians)
    return np.asarray(
        ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )


def _scale(sx: float, sy: float) -> np.ndarray:
    return np.asarray(
        ((sx, 0.0, 0.0), (0.0, sy, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )


def _around(
    point: Iterable[float],
    *,
    angle: float = 0.0,
    sx: float = 1.0,
    sy: float = 1.0,
) -> np.ndarray:
    x, y = point
    return (
        _translation(float(x), float(y))
        @ _rotation(angle)
        @ _scale(sx, sy)
        @ _translation(-float(x), -float(y))
    )


def _transform_point(matrix: np.ndarray, point: Iterable[float]) -> tuple[float, float]:
    x, y = point
    transformed = matrix @ np.asarray((float(x), float(y), 1.0))
    return float(transformed[0]), float(transformed[1])


def _pin_endpoint(
    matrix: np.ndarray,
    target_matrix: np.ndarray,
    endpoint: Iterable[float],
) -> np.ndarray:
    current_x, current_y = _transform_point(matrix, endpoint)
    target_x, target_y = _transform_point(target_matrix, endpoint)
    return _translation(target_x - current_x, target_y - current_y) @ matrix


def _validated_point(
    points: Mapping[str, Sequence[float]], name: str
) -> tuple[float, float]:
    if not isinstance(points, Mapping) or name not in points:
        raise ValueError(f"motion points must contain {name!r}")
    point = points[name]
    if (
        isinstance(point, (str, bytes))
        or not isinstance(point, Sequence)
        or len(point) != 2
    ):
        raise ValueError(f"motion point {name!r} must be an (x, y) pair")
    values: list[float] = []
    for value in point:
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise ValueError(f"motion point {name!r} must be numeric")
        resolved = float(value)
        if not math.isfinite(resolved):
            raise ValueError(f"motion point {name!r} must be finite")
        values.append(resolved)
    return values[0], values[1]


def motion_driver_matrices(
    pose: MotionPose,
    *,
    joints: Mapping[str, Sequence[float]],
    sockets: Mapping[str, Sequence[float]],
    family: int,
    facing: str = "north",
    plant_feet: bool = False,
) -> dict[str, np.ndarray]:
    """Build source-to-destination matrices for any compatible graph rig.

    The public motion program depends only on named pivots, sockets, family,
    and a :class:`MotionPose`.  Procedural specimens and neural owner bindings
    therefore share the exact same curves without sharing or substituting
    pixels.  Clip-wide canvas fitting remains the caller's responsibility.
    """
    if not isinstance(pose, MotionPose):
        raise TypeError("pose must be a MotionPose")
    if isinstance(family, bool) or not isinstance(family, int) or not 0 <= family < 5:
        raise ValueError("family must be an integer in [0, 4]")
    if facing not in FACING_NAMES:
        raise ValueError(f"Unsupported facing {facing!r}; expected one of {FACING_NAMES}")
    if not isinstance(plant_feet, bool):
        raise TypeError("plant_feet must be a bool")

    required_joints = (
        "root",
        "head",
        "left_shoulder",
        "right_shoulder",
        "left_hip",
        "right_hip",
        "appendage_base",
        "weapon_mount",
    )
    resolved_joints = {
        name: _validated_point(joints, name) for name in required_joints
    }
    resolved_sockets = {
        name: _validated_point(sockets, name)
        for name in ("left_foot", "right_foot")
    }
    root = resolved_joints["root"]
    base = (
        _translation(pose.root_dx, pose.root_dy)
        @ _around(
            root,
            angle=pose.global_angle,
            sx=pose.global_sx,
            sy=pose.global_sy,
        )
    )
    body = base @ _around(
        root,
        angle=pose.body_angle,
        sx=pose.body_sx,
        sy=pose.body_sy,
    )
    matrices = {
        "body": body,
        "head": body
        @ _around(
            resolved_joints["head"],
            angle=pose.head_angle,
            sx=pose.head_sx,
            sy=pose.head_sy,
        ),
        "left_arm": body
        @ _around(
            resolved_joints["left_shoulder"], angle=pose.left_arm_angle
        ),
        "right_arm": body
        @ _around(
            resolved_joints["right_shoulder"], angle=pose.right_arm_angle
        ),
        "left_leg": body
        @ _around(resolved_joints["left_hip"], angle=pose.left_leg_angle),
        "right_leg": body
        @ _around(resolved_joints["right_hip"], angle=pose.right_leg_angle),
        "appendage": body
        @ _around(
            resolved_joints["appendage_base"], angle=pose.appendage_angle
        ),
    }
    weapon_parent = matrices["right_arm"] if family == 0 else matrices["head"]
    matrices["weapon"] = weapon_parent @ _around(
        resolved_joints["weapon_mount"], angle=pose.weapon_angle
    )
    if plant_feet:
        matrices["left_leg"] = _pin_endpoint(
            matrices["left_leg"], base, resolved_sockets["left_foot"]
        )
        matrices["right_leg"] = _pin_endpoint(
            matrices["right_leg"], base, resolved_sockets["right_foot"]
        )
    facing_matrix = _around(root, angle=FACING_DEGREES[facing])
    return {name: facing_matrix @ matrix for name, matrix in matrices.items()}


def _smoothstep(value: float) -> float:
    value = min(1.0, max(0.0, value))
    return value * value * (3.0 - 2.0 * value)


def _pose_for(motion: str, phase: float, family: int) -> MotionPose:
    theta = phase * math.tau
    wave = math.sin(theta)
    cosine = math.cos(theta)
    double = math.sin(theta * 2.0)

    if motion == "idle_breathe":
        organic = (1.0, 1.15, 1.35, 1.55, 0.65)[family]
        return MotionPose(
            body_sy=1.0 + 0.055 * organic * wave,
            body_sx=1.0 - 0.025 * organic * wave,
            head_sy=1.0 + 0.035 * organic * wave,
            left_arm_angle=-3.5 * organic * wave,
            right_arm_angle=3.5 * organic * wave,
            appendage_angle=5.0 * organic * wave,
            emission_pulse=1 if wave > 0.55 else 0,
        )
    if motion == "idle_wiggle":
        amplitude = (1.0, 1.25, 1.55, 1.8, 0.7)[family]
        return MotionPose(
            body_angle=4.5 * amplitude * wave,
            head_angle=-6.0 * amplitude * wave,
            left_arm_angle=7.0 * amplitude * wave,
            right_arm_angle=-7.0 * amplitude * wave,
            appendage_angle=13.0 * amplitude * math.sin(theta + 0.45),
            weapon_angle=-4.0 * amplitude * wave,
        )
    if motion == "locomote":
        if family == 0:  # humanoid: counter-swinging arms and planted stride.
            return MotionPose(
                root_dy=-1.2 * abs(wave),
                body_angle=2.0 * wave,
                left_arm_angle=-16.0 * wave,
                right_arm_angle=16.0 * wave,
                left_leg_angle=22.0 * wave,
                right_leg_angle=-22.0 * wave,
                appendage_angle=-10.0 * wave,
            )
        if family == 1:  # animalian: diagonal gait plus active tail.
            return MotionPose(
                root_dy=-0.8 * abs(double),
                body_sy=1.0 - 0.035 * cosine,
                left_arm_angle=20.0 * wave,
                right_arm_angle=-20.0 * wave,
                left_leg_angle=-18.0 * wave,
                right_leg_angle=18.0 * wave,
                appendage_angle=25.0 * math.sin(theta - 0.5),
                head_angle=-4.0 * wave,
            )
        if family == 2:  # plantlike: walking roots with branch counterbalance.
            return MotionPose(
                root_dy=-0.65 * abs(wave),
                body_angle=3.5 * wave,
                left_arm_angle=-10.0 * wave,
                right_arm_angle=10.0 * wave,
                left_leg_angle=17.0 * wave,
                right_leg_angle=-17.0 * wave,
                appendage_angle=22.0 * math.sin(theta + 0.7),
            )
        if family == 3:  # anomaly: hover pulse and asynchronous tendrils.
            return MotionPose(
                root_dy=-1.8 * (1.0 - cosine) * 0.5,
                body_angle=5.0 * double,
                body_sx=1.0 + 0.045 * wave,
                body_sy=1.0 - 0.035 * wave,
                left_arm_angle=18.0 * wave,
                right_arm_angle=-13.0 * math.sin(theta + 0.9),
                left_leg_angle=-12.0 * wave,
                right_leg_angle=15.0 * math.sin(theta + 0.5),
                appendage_angle=32.0 * math.sin(theta - 0.4),
                emission_pulse=1 if cosine < -0.25 else 0,
            )
        # machine: short tread-rock and stabilized turret.
        return MotionPose(
            root_dy=-0.55 * abs(double),
            body_angle=1.8 * wave,
            left_leg_angle=6.0 * wave,
            right_leg_angle=-6.0 * wave,
            left_arm_angle=-4.0 * wave,
            right_arm_angle=4.0 * wave,
            head_angle=-1.8 * wave,
            appendage_angle=8.0 * double,
            emission_pulse=1 if wave > 0.35 else 0,
        )
    if motion == "joy":
        bounce = (1.0 - cosine) * 0.5
        return MotionPose(
            root_dy=-2.0 * bounce,
            body_sy=1.0 + 0.07 * bounce,
            head_sy=1.0 + 0.08 * bounce,
            left_arm_angle=-28.0 - 11.0 * wave,
            right_arm_angle=28.0 + 11.0 * wave,
            left_leg_angle=6.0 * wave,
            right_leg_angle=-6.0 * wave,
            appendage_angle=24.0 * wave,
            emission_pulse=1 if bounce > 0.4 else 0,
        )
    if motion == "anger":
        return MotionPose(
            root_dx=0.8 * math.sin(theta * 3.0),
            body_sx=1.06 + 0.035 * wave,
            body_sy=0.97 - 0.02 * wave,
            head_angle=2.5 * math.sin(theta * 3.0),
            left_arm_angle=20.0 + 8.0 * wave,
            right_arm_angle=-20.0 - 8.0 * wave,
            appendage_angle=-16.0 * wave,
            weapon_angle=7.0 * wave,
            emission_pulse=2 if cosine < -0.1 else 1,
        )
    if motion == "fear":
        tremor = math.sin(theta * 3.0)
        return MotionPose(
            root_dx=1.15 * tremor,
            body_sx=0.92 + 0.025 * cosine,
            body_sy=1.05 - 0.025 * cosine,
            head_sx=1.08,
            head_sy=1.08,
            head_angle=4.0 * tremor,
            left_arm_angle=-14.0 + 5.0 * tremor,
            right_arm_angle=14.0 - 5.0 * tremor,
            left_leg_angle=5.0 * tremor,
            right_leg_angle=-5.0 * tremor,
            appendage_angle=10.0 * tremor,
            emission_pulse=1 if tremor > 0.45 else 0,
        )
    if motion == "confused":
        return MotionPose(
            body_angle=2.5 * wave,
            head_angle=14.0 * wave,
            left_arm_angle=-4.0 * wave,
            right_arm_angle=12.0 + 12.0 * wave,
            appendage_angle=-18.0 * math.sin(theta + 0.5),
            weapon_angle=5.0 * wave,
        )
    if motion == "sleep":
        slow = math.sin(theta)
        return MotionPose(
            body_sy=0.96 + 0.035 * slow,
            head_angle=9.0 + 2.5 * slow,
            head_sy=0.96 + 0.025 * slow,
            left_arm_angle=8.0 + 2.0 * slow,
            right_arm_angle=-8.0 - 2.0 * slow,
            appendage_angle=5.0 * slow,
        )
    if motion == "taunt":
        return MotionPose(
            body_angle=-3.0 * wave,
            head_angle=5.0 * wave,
            left_arm_angle=-8.0 * wave,
            right_arm_angle=20.0 + 28.0 * wave,
            appendage_angle=-20.0 * wave,
            weapon_angle=10.0 * math.sin(theta * 2.0),
            emission_pulse=1 if wave > 0.55 else 0,
        )

    progress = _smoothstep(phase)
    strike = math.sin(math.pi * phase)
    if motion == "attack":
        return MotionPose(
            root_dy=-2.8 * strike,
            global_angle=-4.0 * strike,
            body_sy=1.0 + 0.05 * strike,
            head_angle=-5.0 * strike,
            left_arm_angle=10.0 * strike,
            right_arm_angle=-38.0 + 92.0 * progress,
            left_leg_angle=-8.0 * strike,
            right_leg_angle=8.0 * strike,
            appendage_angle=-20.0 * strike,
            weapon_angle=-24.0 + 50.0 * progress,
            emission_pulse=2 if 0.35 <= phase <= 0.8 else 0,
        )
    if motion == "cast":
        charge = math.sin(math.pi * min(1.0, phase * 1.25))
        return MotionPose(
            root_dy=-1.0 * charge,
            body_sx=1.0 + 0.08 * charge,
            body_sy=1.0 + 0.05 * charge,
            head_sy=1.0 + 0.08 * charge,
            left_arm_angle=-34.0 * progress,
            right_arm_angle=34.0 * progress,
            appendage_angle=38.0 * charge,
            weapon_angle=-14.0 * charge,
            emission_pulse=2 if phase > 0.35 else (1 if phase > 0.12 else 0),
        )
    if motion == "hit":
        recoil = math.sin(math.pi * phase)
        return MotionPose(
            root_dx=2.4 * recoil,
            global_angle=12.0 * recoil,
            body_sx=1.0 - 0.08 * recoil,
            body_sy=1.0 + 0.06 * recoil,
            head_angle=-14.0 * recoil,
            left_arm_angle=-18.0 * recoil,
            right_arm_angle=-22.0 * recoil,
            appendage_angle=28.0 * recoil,
            weapon_angle=-16.0 * recoil,
            emission_pulse=2 if 0.2 <= phase <= 0.55 else 0,
        )
    if motion == "death":
        return MotionPose(
            root_dx=3.0 * progress,
            root_dy=2.0 * progress,
            global_angle=76.0 * progress,
            global_sx=1.0 + 0.08 * progress,
            global_sy=1.0 - 0.14 * progress,
            body_sy=1.0 - 0.12 * progress,
            head_angle=20.0 * progress,
            left_arm_angle=-38.0 * progress,
            right_arm_angle=43.0 * progress,
            left_leg_angle=28.0 * progress,
            right_leg_angle=-24.0 * progress,
            appendage_angle=55.0 * progress,
            weapon_angle=32.0 * progress,
            emission_pulse=1 if 0.15 < phase < 0.58 else 0,
        )
    raise ValueError(f"Unsupported motion: {motion!r}")


def motion_pose(motion: str, phase: float, family: int) -> MotionPose:
    """Return the versioned graph-rig pose for a normalized motion phase."""
    if motion not in MOTION_NAMES:
        raise ValueError(f"Unsupported motion {motion!r}; expected one of {MOTION_NAMES}")
    if isinstance(phase, (bool, np.bool_)) or not isinstance(
        phase, (int, float, np.integer, np.floating)
    ):
        raise ValueError("phase must be numeric and in [0, 1]")
    resolved_phase = float(phase)
    if not math.isfinite(resolved_phase) or not 0.0 <= resolved_phase <= 1.0:
        raise ValueError("phase must be finite and in [0, 1]")
    if isinstance(family, bool) or not isinstance(family, int) or not 0 <= family < 5:
        raise ValueError("family must be an integer in [0, 4]")
    return _pose_for(motion, resolved_phase, family)


def motion_event_specs(
    motion: str,
) -> tuple[tuple[str, float, str | None], ...]:
    """Expose immutable event timing for compatible procedural/neural rigs."""
    if motion not in MOTION_NAMES:
        raise ValueError(f"Unsupported motion {motion!r}; expected one of {MOTION_NAMES}")
    return tuple(_EVENT_SPECS[motion])


def _matrices_for(
    specimen: MorphologySpecimen,
    pose: MotionPose,
    facing: str,
    *,
    plant_feet: bool = False,
) -> dict[str, np.ndarray]:
    return motion_driver_matrices(
        pose,
        joints=specimen.joints,
        sockets=specimen.sockets,
        family=specimen.genome.family,
        facing=facing,
        plant_feet=plant_feet,
    )


def _semantic_owner_drivers(specimen: MorphologySpecimen) -> np.ndarray:
    owners = np.full((CANVAS_SIZE, CANVAS_SIZE), BODY, dtype=np.uint8)
    structural = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=bool)
    for layer_index in _OWNER_PRIORITY:
        mask = specimen.layers[layer_index] > 0
        owners[mask] = layer_index
        structural |= mask
    owners[~structural] = BODY
    return owners


def _transformed_bounds(
    specimen: MorphologySpecimen,
    matrices: dict[str, np.ndarray],
) -> tuple[float, float, float, float]:
    minimum_x = float("inf")
    minimum_y = float("inf")
    maximum_x = float("-inf")
    maximum_y = float("-inf")
    for layer_index in STRUCTURAL_LAYERS:
        points = np.argwhere(specimen.layers[layer_index] > 0)
        if not len(points):
            continue
        homogeneous = np.stack(
            (
                points[:, 1].astype(np.float64),
                points[:, 0].astype(np.float64),
                np.ones(len(points), dtype=np.float64),
            ),
            axis=0,
        )
        transformed = matrices[_DRIVER_BY_LAYER[layer_index]] @ homogeneous
        minimum_x = min(minimum_x, float(transformed[0].min()))
        minimum_y = min(minimum_y, float(transformed[1].min()))
        maximum_x = max(maximum_x, float(transformed[0].max()))
        maximum_y = max(maximum_y, float(transformed[1].max()))
    return minimum_x, minimum_y, maximum_x, maximum_y


def _fit_matrix(
    specimen: MorphologySpecimen,
    matrix_sets: Iterable[dict[str, np.ndarray]],
) -> np.ndarray:
    bounds = [_transformed_bounds(specimen, matrices) for matrices in matrix_sets]
    min_x = min(bound[0] for bound in bounds)
    min_y = min(bound[1] for bound in bounds)
    max_x = max(bound[2] for bound in bounds)
    max_y = max(bound[3] for bound in bounds)
    # One extra pixel beyond the semantic safety margin is reserved for
    # nearest-neighbour rounding and the RGBA outline.
    safe_min = float(SAFETY_MARGIN + 1)
    safe_max = float(CANVAS_SIZE - SAFETY_MARGIN - 2)
    safe_span = safe_max - safe_min
    span_x = max(1.0, max_x - min_x)
    span_y = max(1.0, max_y - min_y)
    scale = min(1.0, safe_span / span_x, safe_span / span_y)

    root_x, root_y = map(float, specimen.joints["root"])
    desired_tx = root_x - scale * root_x
    desired_ty = root_y - scale * root_y
    minimum_tx = safe_min - scale * min_x
    maximum_tx = safe_max - scale * max_x
    minimum_ty = safe_min - scale * min_y
    maximum_ty = safe_max - scale * max_y
    tx = min(max(desired_tx, minimum_tx), maximum_tx)
    ty = min(max(desired_ty, minimum_ty), maximum_ty)
    return _translation(tx, ty) @ _scale(scale, scale)


def _warp_mask(mask: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if not bool(mask.any()):
        return np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)
    inverse = np.linalg.inv(matrix)
    coefficients = tuple(float(value) for value in inverse[:2].reshape(-1))
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    transformed = image.transform(
        (CANVAS_SIZE, CANVAS_SIZE),
        Image.Transform.AFFINE,
        coefficients,
        resample=Image.Resampling.NEAREST,
        fillcolor=0,
    )
    result = (np.asarray(transformed, dtype=np.uint8) > 0).astype(np.uint8)
    if not bool(result.any()):
        # A one-pixel semantic marker can vanish under a strong diagonal
        # resample. Preserve it by forwarding the source pixel centers.
        points = np.argwhere(mask > 0)
        homogeneous = np.stack(
            (
                points[:, 1].astype(np.float64),
                points[:, 0].astype(np.float64),
                np.ones(len(points), dtype=np.float64),
            ),
            axis=0,
        )
        forwarded = matrix @ homogeneous
        xs = np.clip(np.rint(forwarded[0]).astype(np.int64), 0, CANVAS_SIZE - 1)
        ys = np.clip(np.rint(forwarded[1]).astype(np.int64), 0, CANVAS_SIZE - 1)
        result[ys, xs] = 1
    return result


def _nearest_pixel(mask: np.ndarray, target: tuple[float, float]) -> list[int]:
    points = np.argwhere(mask > 0)
    if not len(points):
        raise ValueError("Cannot snap an anchor to an empty semantic layer")
    tx, ty = target
    distance = (points[:, 1] - tx) ** 2 + (points[:, 0] - ty) ** 2
    index = int(np.argmin(distance))
    return [int(points[index, 1]), int(points[index, 0])]


def _dilate(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(bool), 1, mode="constant")
    return np.logical_or.reduce(
        [
            padded[y : y + mask.shape[0], x : x + mask.shape[1]]
            for y in range(3)
            for x in range(3)
        ]
    )


def _attached(child: np.ndarray, parent: np.ndarray) -> bool:
    return bool((child.astype(bool) & _dilate(parent)).any())


def _closest_pair(first: np.ndarray, second: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    first_points = np.argwhere(first > 0)
    second_points = np.argwhere(second > 0)
    if not len(first_points) or not len(second_points):
        raise ValueError("Cannot connect empty semantic masks")
    best_distance = float("inf")
    best: tuple[tuple[int, int], tuple[int, int]] | None = None
    for start in range(0, len(first_points), 64):
        chunk = first_points[start : start + 64]
        delta = chunk[:, None, :] - second_points[None, :, :]
        distances = np.sum(delta * delta, axis=2)
        flat_index = int(np.argmin(distances))
        row, column = np.unravel_index(flat_index, distances.shape)
        distance = float(distances[row, column])
        if distance < best_distance:
            first_y, first_x = map(int, chunk[row])
            second_y, second_x = map(int, second_points[column])
            best_distance = distance
            best = ((first_x, first_y), (second_x, second_y))
    if best is None:
        raise AssertionError("closest-pair search produced no candidate")
    return best


def _connect_layer(
    layers: np.ndarray,
    child_index: int,
    parent_indices: tuple[int, ...],
    width: int,
) -> None:
    child = layers[child_index] > 0
    parent = np.logical_or.reduce(layers[list(parent_indices)] > 0)
    if _attached(child, parent):
        return
    child_point, parent_point = _closest_pair(child, parent)
    image = Image.fromarray((layers[child_index] * 255).astype(np.uint8))
    ImageDraw.Draw(image).line(
        (child_point, parent_point), fill=255, width=max(1, min(3, width))
    )
    layers[child_index] = (np.asarray(image, dtype=np.uint8) > 0).astype(np.uint8)


def _repair_topology(layers: np.ndarray, limb_width: int) -> None:
    attachments = (
        (HEAD, (BODY,)),
        (LEFT_ARM, (BODY,)),
        (RIGHT_ARM, (BODY,)),
        (LEFT_LEG, (BODY,)),
        (RIGHT_LEG, (BODY,)),
        (APPENDAGE, (BODY, HEAD)),
        (WEAPON, (BODY, HEAD, RIGHT_ARM)),
    )
    for child_index, parent_indices in attachments:
        _connect_layer(layers, child_index, parent_indices, limb_width)
    _connect_structural_union(layers, limb_width)


def _component_masks(mask: np.ndarray) -> list[np.ndarray]:
    active = mask.astype(bool)
    seen = np.zeros_like(active)
    components: list[np.ndarray] = []
    height, width = active.shape
    for start_y in range(height):
        for start_x in range(width):
            if not active[start_y, start_x] or seen[start_y, start_x]:
                continue
            component = np.zeros_like(active)
            stack = [(start_y, start_x)]
            seen[start_y, start_x] = True
            component[start_y, start_x] = True
            while stack:
                y, x = stack.pop()
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if (
                            0 <= ny < height
                            and 0 <= nx < width
                            and active[ny, nx]
                            and not seen[ny, nx]
                        ):
                            seen[ny, nx] = True
                            component[ny, nx] = True
                            stack.append((ny, nx))
            components.append(component)
    return components


def _connect_structural_union(layers: np.ndarray, limb_width: int) -> None:
    """Reconnect raster islands introduced by diagonal nearest resampling.

    The source graph is connected. If a one-pixel branch loses a diagonal
    bridge during rotation, this restores the missing graph edge on the same
    semantic owner rather than dilating the whole sprite.
    """
    for _ in range(len(STRUCTURAL_LAYERS)):
        structural = np.logical_or.reduce(layers[list(STRUCTURAL_LAYERS)] > 0)
        components = _component_masks(structural)
        if len(components) <= 1:
            return
        components.sort(key=lambda values: int(values.sum()), reverse=True)
        connected = components[0]
        for island in components[1:]:
            island_point, connected_point = _closest_pair(island, connected)
            ownership = [
                int(np.logical_and(layers[index] > 0, island).sum())
                for index in STRUCTURAL_LAYERS
            ]
            owner_index = STRUCTURAL_LAYERS[int(np.argmax(ownership))]
            image = Image.fromarray((layers[owner_index] * 255).astype(np.uint8))
            ImageDraw.Draw(image).line(
                (island_point, connected_point),
                fill=255,
                width=max(1, min(2, limb_width)),
            )
            layers[owner_index] = (
                np.asarray(image, dtype=np.uint8) > 0
            ).astype(np.uint8)
            connected |= island
            bridge = np.asarray(image, dtype=np.uint8) > 0
            connected |= bridge
    structural = np.logical_or.reduce(layers[list(STRUCTURAL_LAYERS)] > 0)
    if component_count(structural) != 1:
        raise ValueError("Unable to restore connected structural topology")


def _expression_masks(
    specimen: MorphologySpecimen,
    motion: str,
    phase: float,
) -> tuple[np.ndarray, np.ndarray]:
    detail = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)
    emission = np.zeros_like(detail)
    if motion not in {"joy", "anger", "fear", "confused", "sleep", "taunt"}:
        return detail, emission
    head = specimen.layers[HEAD] > 0
    points = np.argwhere(head)
    center_x = float(points[:, 1].mean())
    center_y = float(points[:, 0].mean())

    def mark(target_x: float, target_y: float, *, glow: bool = False) -> None:
        point = _nearest_pixel(head, (target_x, target_y))
        detail[point[1], point[0]] = 1
        if glow:
            emission[point[1], point[0]] = 1

    eye_y = center_y - 1.0
    if motion == "joy":
        mark(center_x - 2.0, eye_y, glow=True)
        mark(center_x + 2.0, eye_y, glow=True)
        for offset in (-1.0, 0.0, 1.0):
            mark(center_x + offset, center_y + 2.0)
    elif motion == "anger":
        mark(center_x - 2.0, eye_y - 1.0, glow=True)
        mark(center_x - 1.0, eye_y, glow=True)
        mark(center_x + 1.0, eye_y, glow=True)
        mark(center_x + 2.0, eye_y - 1.0, glow=True)
    elif motion == "fear":
        mark(center_x - 2.0, eye_y, glow=True)
        mark(center_x + 2.0, eye_y, glow=True)
        mark(center_x, center_y + 2.0, glow=math.sin(phase * math.tau * 3.0) > 0.0)
    elif motion == "confused":
        mark(center_x - 2.0, eye_y, glow=True)
        mark(center_x + 2.0, eye_y - 1.0)
        mark(center_x + 1.0, center_y + 2.0)
    elif motion == "sleep":
        for x_offset in (-2.0, -1.0, 1.0, 2.0):
            mark(center_x + x_offset, eye_y)
    elif motion == "taunt":
        mark(center_x - 2.0, eye_y, glow=True)
        mark(center_x + 2.0, eye_y)
        mark(center_x + 1.0, center_y + 2.0)
        mark(center_x + 2.0, center_y + 1.0)
    return detail, emission


def _apply_emission_pulse(layers: np.ndarray, strength: int) -> None:
    if strength <= 0:
        return
    structural = np.logical_or.reduce(layers[list(STRUCTURAL_LAYERS)] > 0)
    emission = layers[EMISSION] > 0
    structural_pixels = int(structural.sum())
    # Role-conditioned source masks may already be heavily emissive. A naive
    # dilation would turn them into a flat white silhouette, erasing readable
    # anatomy. Grow only a deterministic, coverage-capped subset of the rim.
    maximum_fraction = 0.34 if strength == 1 else 0.46
    maximum_pixels = max(int(emission.sum()), int(structural_pixels * maximum_fraction))
    for iteration in range(min(2, strength)):
        candidates = _dilate(emission) & structural & ~emission
        capacity = maximum_pixels - int(emission.sum())
        if capacity <= 0 or not candidates.any():
            break
        points = np.argwhere(candidates)
        keys = (
            points[:, 1].astype(np.uint64) * np.uint64(73_856_093)
            ^ points[:, 0].astype(np.uint64) * np.uint64(19_349_663)
            ^ np.uint64((iteration + 1) * 83_492_791)
        )
        selected = points[np.argsort(keys, kind="stable")[:capacity]]
        emission[selected[:, 0], selected[:, 1]] = True
    layers[EMISSION] = emission.astype(np.uint8)
    layers[DETAIL] |= layers[EMISSION]


def _render_layers(
    specimen: MorphologySpecimen,
    matrices: dict[str, np.ndarray],
    pose: MotionPose,
    motion: str,
    phase: float,
) -> np.ndarray:
    layers = np.zeros_like(specimen.layers)
    for layer_index, driver in _DRIVER_BY_LAYER.items():
        layers[layer_index] = _warp_mask(
            specimen.layers[layer_index], matrices[driver]
        )

    owners = _semantic_owner_drivers(specimen)
    expression_detail, expression_emission = _expression_masks(
        specimen, motion, phase
    )
    source_detail = np.logical_or(specimen.layers[DETAIL] > 0, expression_detail > 0)
    source_emission = np.logical_or(
        specimen.layers[EMISSION] > 0, expression_emission > 0
    )
    for owner_index in _OWNER_PRIORITY:
        driver = _DRIVER_BY_LAYER[owner_index]
        owner = owners == owner_index
        layers[DETAIL] |= _warp_mask(source_detail & owner, matrices[driver])
        layers[EMISSION] |= _warp_mask(source_emission & owner, matrices[driver])

    _repair_topology(layers, specimen.genome.limb_thickness)
    structural = np.logical_or.reduce(layers[list(STRUCTURAL_LAYERS)] > 0)
    layers[DETAIL] &= structural.astype(np.uint8)
    layers[EMISSION] &= structural.astype(np.uint8)
    _apply_emission_pulse(layers, pose.emission_pulse)
    # Fit calculations reserve this border. Clearing it is a defensive
    # assertion boundary, not a clipping mechanism; validation verifies that
    # topology remains intact afterward.
    layers[:, :SAFETY_MARGIN] = 0
    layers[:, -SAFETY_MARGIN:] = 0
    layers[:, :, :SAFETY_MARGIN] = 0
    layers[:, :, -SAFETY_MARGIN:] = 0
    return layers.astype(np.uint8)


def _anchors_for(
    specimen: MorphologySpecimen,
    layers: np.ndarray,
    matrices: dict[str, np.ndarray],
) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    joints: dict[str, list[int]] = {}
    sockets: dict[str, list[int]] = {}
    for name, layer_index in JOINT_LAYER.items():
        target = _transform_point(matrices[_POINT_DRIVER[name]], specimen.joints[name])
        joints[name] = _nearest_pixel(layers[layer_index], target)
    for name, layer_index in SOCKET_LAYER.items():
        target = _transform_point(matrices[_POINT_DRIVER[name]], specimen.sockets[name])
        sockets[name] = _nearest_pixel(layers[layer_index], target)
    return joints, sockets


def _frame_hash(
    index: int,
    phase: float,
    layers: np.ndarray,
    tokens: np.ndarray,
    rgba: np.ndarray,
    joints: dict[str, list[int]],
    sockets: dict[str, list[int]],
) -> str:
    digest = hashlib.sha256()
    digest.update(index.to_bytes(4, "little", signed=False))
    digest.update(f"{phase:.9f}".encode("ascii"))
    for array in (layers, tokens, rgba):
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    digest.update(
        json.dumps(
            {"joints": joints, "sockets": sockets},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return digest.hexdigest()


def _render_frame(
    specimen: MorphologySpecimen,
    motion: str,
    index: int,
    phase: float,
    pose: MotionPose,
    matrices: dict[str, np.ndarray],
) -> MotionFrame:
    layers = _render_layers(specimen, matrices, pose, motion, phase)
    tokens = layers_to_tokens(layers)
    joints, sockets = _anchors_for(specimen, layers, matrices)
    rgba = compose_rgba(layers, specimen.palette)
    sha256 = _frame_hash(index, phase, layers, tokens, rgba, joints, sockets)
    return MotionFrame(
        index=index,
        phase=phase,
        layers=layers,
        tokens=tokens,
        rgba=rgba,
        joints=joints,
        sockets=sockets,
        sha256=sha256,
    )


def _stabilize_stance_sockets(
    frames: tuple[MotionFrame, ...],
) -> tuple[MotionFrame, ...]:
    """Choose one raster-valid foot anchor shared by every planted frame.

    Continuous endpoint pinning can land exactly between two pixel centers.
    Nearest-neighbour rasterization may then expose alternating but equivalent
    edge pixels. Selecting the closest pixel in the per-clip mask intersection
    keeps the gameplay socket bit-stable without weakening validation.
    """
    stable: dict[str, list[int]] = {}
    for socket_name, layer_index in (
        ("left_foot", LEFT_LEG),
        ("right_foot", RIGHT_LEG),
    ):
        common = np.logical_and.reduce(
            [frame.layers[layer_index].astype(bool) for frame in frames]
        )
        if not common.any():
            raise ValueError(
                f"No shared semantic pixel is available for planted {socket_name}"
            )
        mean_x = float(np.mean([frame.sockets[socket_name][0] for frame in frames]))
        mean_y = float(np.mean([frame.sockets[socket_name][1] for frame in frames]))
        stable[socket_name] = _nearest_pixel(common, (mean_x, mean_y))

    result: list[MotionFrame] = []
    for frame in frames:
        sockets = {name: list(value) for name, value in frame.sockets.items()}
        sockets.update({name: list(value) for name, value in stable.items()})
        sha256 = _frame_hash(
            frame.index,
            frame.phase,
            frame.layers,
            frame.tokens,
            frame.rgba,
            frame.joints,
            sockets,
        )
        result.append(replace(frame, sockets=sockets, sha256=sha256))
    return tuple(result)


def _span(points: Iterable[list[int]]) -> list[int]:
    values = list(points)
    xs = [point[0] for point in values]
    ys = [point[1] for point in values]
    return [max(xs) - min(xs), max(ys) - min(ys)]


def _clip_manifest(
    specimen: MorphologySpecimen,
    motion: str,
    facing: str,
    fps: int,
    loop: bool,
    frames: tuple[MotionFrame, ...],
) -> dict[str, Any]:
    semantic_payloads = {
        hashlib.sha256(frame.layers.tobytes() + frame.tokens.tobytes()).hexdigest()
        for frame in frames
    }
    changed = []
    first_visible = np.logical_or.reduce(frames[0].layers > 0)
    for frame in frames[1:]:
        visible = np.logical_or.reduce(frame.layers > 0)
        union = first_visible | visible
        changed.append(float(np.logical_xor(first_visible, visible).sum()) / max(1, int(union.sum())))
    root_span = _span(frame.joints["root"] for frame in frames)
    left_foot_span = _span(frame.sockets["left_foot"] for frame in frames)
    right_foot_span = _span(frame.sockets["right_foot"] for frame in frames)
    base = {
        "format": MOTION_FORMAT,
        "id": f"{specimen.manifest['id']}__{motion}__{facing}",
        "source_id": specimen.manifest["id"],
        "source_semantic_sha256": specimen.manifest["hashes"]["semantic_sha256"],
        "motion_renderer_version": MOTION_RENDERER_VERSION,
        "family": specimen.genome.family_name,
        "motion": motion,
        "facing": facing,
        "loop": loop,
        "fps": fps,
        "frame_count": len(frames),
        "canvas_size": [CANVAS_SIZE, CANVAS_SIZE],
        "safety_margin": SAFETY_MARGIN,
        "layer_names": list(LAYER_NAMES),
        "joint_names": list(JOINT_LAYER),
        "socket_names": list(SOCKET_LAYER),
        "events": [
            {
                "name": name,
                "frame": min(len(frames) - 1, int(round(phase * (len(frames) - 1)))),
                "phase": phase,
                "socket": socket,
            }
            for name, phase, socket in _EVENT_SPECS[motion]
        ],
        "frame_sha256": [frame.sha256 for frame in frames],
        "metrics": {
            "unique_semantic_frames": len(semantic_payloads),
            "max_changed_pixel_fraction": round(max(changed, default=0.0), 7),
            "root_span": root_span,
            "left_foot_span": left_foot_span,
            "right_foot_span": right_foot_span,
            "max_structural_components": max(
                component_count(
                    np.logical_or.reduce(frame.layers[list(STRUCTURAL_LAYERS)] > 0)
                )
                for frame in frames
            ),
            "margin_clear": True,
            "field_tuples_valid": True,
        },
    }
    digest = hashlib.sha256(
        json.dumps(base, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    base["hashes"] = {"clip_sha256": digest}
    return base


def generate_motion_clip(
    specimen: MorphologySpecimen,
    motion: str,
    *,
    facing: str = "north",
    frame_count: int | None = None,
    fps: int | None = None,
) -> MotionClip:
    if not isinstance(specimen, MorphologySpecimen):
        raise TypeError("specimen must be a MorphologySpecimen")
    specimen_errors = validate_specimen(specimen)
    if specimen_errors:
        raise ValueError("Invalid source specimen: " + "; ".join(specimen_errors))
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
    if isinstance(resolved_fps, bool) or not isinstance(resolved_fps, int) or not 1 <= resolved_fps <= 60:
        raise ValueError("fps must be an integer in [1, 60]")

    phases = [index / float(resolved_count - 1) for index in range(resolved_count)]
    poses = [_pose_for(motion, phase, specimen.genome.family) for phase in phases]
    if motion in LOOPING_MOTIONS:
        poses[-1] = poses[0]
    raw_matrices = [
        _matrices_for(
            specimen,
            pose,
            facing,
            plant_feet=motion in STABLE_STANCE_MOTIONS,
        )
        for pose in poses
    ]
    fit = _fit_matrix(specimen, raw_matrices)
    fitted_matrices = [
        {name: fit @ matrix for name, matrix in matrices.items()}
        for matrices in raw_matrices
    ]
    frames = tuple(
        _render_frame(
            specimen,
            motion,
            index,
            phases[index],
            poses[index],
            fitted_matrices[index],
        )
        for index in range(resolved_count)
    )
    if motion in STABLE_STANCE_MOTIONS:
        frames = _stabilize_stance_sockets(frames)
    manifest = _clip_manifest(
        specimen,
        motion,
        facing,
        resolved_fps,
        motion in LOOPING_MOTIONS,
        frames,
    )
    clip = MotionClip(
        specimen=specimen,
        motion=motion,
        facing=facing,
        fps=resolved_fps,
        loop=motion in LOOPING_MOTIONS,
        frames=frames,
        manifest=manifest,
    )
    errors = validate_motion_clip(clip)
    if errors:
        raise ValueError("; ".join(errors))
    return clip


@lru_cache(maxsize=1)
def allowed_training_field_tuples() -> frozenset[tuple[int, int, int]]:
    """Cross-field tuples possible under the versioned semantic rules.

    This is generated from the field-writing precedence itself rather than
    learned from a finite sample, so animated overlap cannot create a tuple
    that was merely absent from a small training split.
    """
    allowed: set[tuple[int, int, int]] = {(0, 0, 0), (16, 9, 3)}
    structural_indices = set(STRUCTURAL_LAYERS)
    family_materials = (1, 2, 3, 7, 4)
    for bits in range(1, 1 << len(LAYER_NAMES)):
        present = {index for index in range(len(LAYER_NAMES)) if bits & (1 << index)}
        if not (present & structural_indices):
            continue
        owner = max(present) + 1
        if CORE in present:
            owner = CORE + 1
        for base_material in family_materials:
            material = base_material
            if ARMOR in present:
                material = 5
            if WEAPON in present:
                material = 6
            if CORE in present:
                material = 7
            if DETAIL in present:
                material = 8
            if EMISSION in present:
                material = 9
            emission_level = 0
            if DETAIL in present:
                emission_level = 1
            if EMISSION in present:
                emission_level = 2
            if CORE in present:
                emission_level = 3
            allowed.add((owner, material, emission_level))
            allowed.add((13, material, emission_level))
            allowed.add((14, material, emission_level))
            if DETAIL in present and EMISSION not in present:
                allowed.add((15, material, emission_level))
    return frozenset(allowed)


def frame_training_tuples(
    frame: MotionFrame,
    specimen: MorphologySpecimen,
) -> frozenset[tuple[int, int, int]]:
    fields = frame.training_fields(specimen)
    triples = np.stack(
        (fields.part_owner, fields.material, fields.emission_level), axis=-1
    ).reshape(-1, 3)
    return frozenset(tuple(map(int, row)) for row in triples)


def _point_errors(
    points: dict[str, list[int]],
    expected: dict[str, int],
    layers: np.ndarray,
    label: str,
) -> list[str]:
    errors: list[str] = []
    if set(points) != set(expected):
        return [f"{label} keys disagree with the rig contract"]
    for name, layer_index in expected.items():
        value = points[name]
        if (
            not isinstance(value, list)
            or len(value) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
        ):
            errors.append(f"{label}.{name} is not an integer point")
            continue
        x, y = value
        if not (0 <= x < CANVAS_SIZE and 0 <= y < CANVAS_SIZE):
            errors.append(f"{label}.{name} is outside the canvas")
        elif not layers[layer_index, y, x]:
            errors.append(f"{label}.{name} does not land on {LAYER_NAMES[layer_index]}")
    return errors


def validate_motion_frame(
    frame: MotionFrame,
    specimen: MorphologySpecimen,
) -> list[str]:
    errors: list[str] = []
    expected_layers = (len(LAYER_NAMES), CANVAS_SIZE, CANVAS_SIZE)
    if frame.layers.shape != expected_layers or frame.layers.dtype != np.uint8:
        return [f"layers must be uint8 {expected_layers}"]
    if not np.isin(frame.layers, (0, 1)).all():
        errors.append("layers must be binary")
    if any(int(layer.sum()) == 0 for layer in frame.layers):
        errors.append("every semantic layer must remain nonempty")
    if frame.tokens.shape != (CANVAS_SIZE, CANVAS_SIZE) or frame.tokens.dtype != np.uint8:
        errors.append("tokens must be uint8 48x48")
    elif not np.array_equal(frame.tokens, layers_to_tokens(frame.layers)):
        errors.append("tokens disagree with semantic layers")
    if frame.rgba.shape != (CANVAS_SIZE, CANVAS_SIZE, 4) or frame.rgba.dtype != np.uint8:
        errors.append("rgba must be uint8 48x48x4")
    elif not np.array_equal(frame.rgba, compose_rgba(frame.layers, specimen.palette)):
        errors.append("rgba disagrees with semantic layers and palette")
    visible = np.logical_or.reduce(frame.layers > 0)
    if (
        visible[:SAFETY_MARGIN].any()
        or visible[-SAFETY_MARGIN:].any()
        or visible[:, :SAFETY_MARGIN].any()
        or visible[:, -SAFETY_MARGIN:].any()
    ):
        errors.append("semantic pixels violate the safety margin")
    structural = np.logical_or.reduce(frame.layers[list(STRUCTURAL_LAYERS)] > 0)
    components = component_count(structural)
    if components != 1:
        errors.append(f"structural union has {components} components")
    attachments = (
        (HEAD, (BODY,)),
        (LEFT_ARM, (BODY,)),
        (RIGHT_ARM, (BODY,)),
        (LEFT_LEG, (BODY,)),
        (RIGHT_LEG, (BODY,)),
        (APPENDAGE, (BODY, HEAD)),
        (WEAPON, (BODY, HEAD, RIGHT_ARM)),
    )
    for child_index, parents in attachments:
        parent = np.logical_or.reduce(frame.layers[list(parents)] > 0)
        if not _attached(frame.layers[child_index], parent):
            errors.append(f"{LAYER_NAMES[child_index]} is detached")
    if bool((frame.layers[[DETAIL, EMISSION]] > structural).any()):
        errors.append("detail or emission escapes the structural silhouette")
    errors.extend(_point_errors(frame.joints, JOINT_LAYER, frame.layers, "joints"))
    errors.extend(_point_errors(frame.sockets, SOCKET_LAYER, frame.layers, "sockets"))
    try:
        tuples = frame_training_tuples(frame, specimen)
        invalid = tuples - allowed_training_field_tuples()
        if invalid:
            errors.append(f"invalid part/material/emission tuples: {sorted(invalid)}")
    except ValueError as error:
        errors.append(f"training fields: {error}")
    expected_hash = _frame_hash(
        frame.index,
        frame.phase,
        frame.layers,
        frame.tokens,
        frame.rgba,
        frame.joints,
        frame.sockets,
    )
    if frame.sha256 != expected_hash:
        errors.append("frame hash is incorrect")
    return errors


def validate_motion_clip(clip: MotionClip) -> list[str]:
    errors: list[str] = []
    if clip.motion not in MOTION_NAMES:
        errors.append("motion is outside the motion vocabulary")
    if clip.facing not in FACING_NAMES:
        errors.append("facing is outside the direction vocabulary")
    if clip.loop != (clip.motion in LOOPING_MOTIONS):
        errors.append("loop flag disagrees with motion contract")
    if len(clip.frames) < 3:
        errors.append("motion clip has fewer than three frames")
    for index, frame in enumerate(clip.frames):
        if frame.index != index:
            errors.append(f"frame {index} has incorrect index")
        errors.extend(f"frame {index}: {error}" for error in validate_motion_frame(frame, clip.specimen))
    if clip.loop and clip.frames:
        first, last = clip.frames[0], clip.frames[-1]
        if not (
            np.array_equal(first.layers, last.layers)
            and np.array_equal(first.tokens, last.tokens)
            and np.array_equal(first.rgba, last.rgba)
            and first.joints == last.joints
            and first.sockets == last.sockets
        ):
            errors.append("loop endpoints are not bit-exact")
    semantic_hashes = {
        hashlib.sha256(frame.layers.tobytes() + frame.tokens.tobytes()).hexdigest()
        for frame in clip.frames
    }
    if len(semantic_hashes) < 2:
        errors.append("motion has no nonzero semantic amplitude")
    if clip.motion in STABLE_STANCE_MOTIONS:
        for socket_name in ("left_foot", "right_foot"):
            if len({tuple(frame.sockets[socket_name]) for frame in clip.frames}) != 1:
                errors.append(f"{socket_name} is not stable during {clip.motion}")
    expected_manifest = _clip_manifest(
        clip.specimen,
        clip.motion,
        clip.facing,
        clip.fps,
        clip.loop,
        clip.frames,
    )
    if clip.manifest != expected_manifest:
        errors.append("clip manifest is incomplete or inconsistent")
    if clip.manifest.get("metrics", {}).get("max_structural_components") != 1:
        errors.append("clip manifest does not certify connected topology")
    try:
        json.dumps(clip.manifest, allow_nan=False)
    except (TypeError, ValueError) as error:
        errors.append(f"clip manifest is not strict JSON: {error}")
    return errors


def assert_valid_motion_clip(clip: MotionClip) -> None:
    errors = validate_motion_clip(clip)
    if errors:
        raise ValueError("; ".join(errors))
