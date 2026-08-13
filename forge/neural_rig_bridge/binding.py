from __future__ import annotations

from dataclasses import replace
import hashlib
import io
from itertools import islice
import json
import math
from pathlib import Path, PurePosixPath
import re
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence
import zipfile

from jsonschema import Draft202012Validator
import numpy as np

from ..morphology.constants import (
    CANVAS_SIZE,
    EMISSION_LEVEL_NAMES,
    FAMILIES,
    GUIDE_CHANNEL_NAMES,
    MATERIAL_NAMES,
    PART_OWNER_NAMES,
    ROLE_NAMES,
    SAFETY_MARGIN,
    SUBTYPE_NAMES,
)
from .hashing import (
    aligned_fields_hash,
    array_hash,
    binder_source_hash,
    canonical_json_hash,
    evaluator_tuple_fingerprint,
    owner_tuple_hash,
    tuple_fingerprint,
)
from .model import (
    ADAPTER_FORMAT,
    BACKGROUND_DRIVER,
    BINDER_VERSION,
    BINDING_FORMAT,
    DRIVER_INDEX,
    DRIVER_NAMES,
    JOINT_DRIVER,
    SOCKET_DRIVER,
    BindingRejected,
    DerivedAnatomy,
    NeuralRigBinding,
    MIN_DRIVER_PIXELS,
    OwnerLayerBinding,
    RigAnchor,
    readonly_array,
)


_SAMPLE_ID = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$")
_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_UPSTREAM_NAME = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")
_RAW_SAMPLE_FORMAT = "nullvector-multifield-raw-sample-v1"
_RAW_VALIDATION_FORMAT = "nullvector-multifield-generation-validation-v2"
_RAW_MANIFEST_MAX_BYTES = 2 * 1024 * 1024
_RAW_ARCHIVE_MAX_BYTES = 4 * 1024 * 1024
_RAW_ARCHIVE_MAX_UNCOMPRESSED_BYTES = 4 * 1024 * 1024
_MAX_LEGAL_TUPLES = (
    len(PART_OWNER_NAMES) * len(MATERIAL_NAMES) * len(EMISSION_LEVEL_NAMES)
)
_ANATOMY_SOURCES = {
    "conditioned_anatomy_points_v1",
    "guide_owner_geometry-v1",
}

_DIRECT_OWNER_DRIVER = {
    1: "body",
    2: "body",
    3: "head",
    4: "left_arm",
    5: "right_arm",
    6: "left_leg",
    7: "right_leg",
    8: "appendage",
    9: "weapon",
    10: "body",
}

_JOINT_OWNER = {
    "root": 1,
    "head": 3,
    "left_shoulder": 4,
    "right_shoulder": 5,
    "left_hip": 6,
    "right_hip": 7,
    "appendage_base": 8,
    "weapon_mount": 9,
}

_SOCKET_OWNER = {
    "focus": 10,
    "muzzle": 9,
    "left_hand": 4,
    "right_hand": 5,
    "left_foot": 6,
    "right_foot": 7,
    "appendage_tip": 8,
}

_DRIVER_SEGMENT = {
    "body": (("joint", "root"), ("socket", "focus")),
    "head": (("joint", "head"), ("joint", "head")),
    "left_arm": (("joint", "left_shoulder"), ("socket", "left_hand")),
    "right_arm": (("joint", "right_shoulder"), ("socket", "right_hand")),
    "left_leg": (("joint", "left_hip"), ("socket", "left_foot")),
    "right_leg": (("joint", "right_hip"), ("socket", "right_foot")),
    "appendage": (("joint", "appendage_base"), ("socket", "appendage_tip")),
    "weapon": (("joint", "weapon_mount"), ("socket", "muzzle")),
}


def _family_index(family: int | str) -> int:
    if isinstance(family, bool):
        raise BindingRejected(["family must be a family name or integer index"])
    if isinstance(family, str):
        if family not in FAMILIES:
            raise BindingRejected([f"unsupported family {family!r}"])
        return FAMILIES.index(family)
    if isinstance(family, (int, np.integer)) and 0 <= int(family) < len(FAMILIES):
        return int(family)
    raise BindingRejected(["family index must be in [0, 4]"])


def _normalize_legal_tuples(
    legal_tuples: np.ndarray | Iterable[tuple[int, int, int]] | None,
) -> np.ndarray:
    if legal_tuples is None:
        # Imported lazily: the bridge uses the compiler's versioned semantic
        # vocabulary but never calls its pixel renderer.
        from ..morphology.motion import allowed_training_field_tuples

        values = np.asarray(sorted(allowed_training_field_tuples()), dtype=np.uint8)
    elif isinstance(legal_tuples, np.ndarray):
        values = np.asarray(legal_tuples)
    else:
        try:
            collected = list(islice(iter(legal_tuples), _MAX_LEGAL_TUPLES + 1))
        except TypeError as error:
            raise BindingRejected(["legal_tuples must be an iterable table"]) from error
        if len(collected) > _MAX_LEGAL_TUPLES:
            raise BindingRejected(
                [f"legal_tuples exceeds the {_MAX_LEGAL_TUPLES}-row semantic bound"]
            )
        try:
            values = np.asarray(collected)
        except ValueError as error:
            raise BindingRejected(["legal_tuples must be a rectangular [N, 3] table"]) from error
    if values.ndim != 2 or values.shape[1:] != (3,) or not len(values):
        raise BindingRejected(["legal_tuples must be a nonempty [N, 3] table"])
    if len(values) > _MAX_LEGAL_TUPLES:
        raise BindingRejected(
            [f"legal_tuples exceeds the {_MAX_LEGAL_TUPLES}-row semantic bound"]
        )
    if values.dtype.kind not in "ui":
        raise BindingRejected(["legal_tuples must contain integers"])
    if int(values.min()) < 0 or int(values.max()) > 255:
        raise BindingRejected(["legal_tuples values must fit uint8"])
    normalized = np.unique(values.astype(np.uint8), axis=0)
    order = np.lexsort((normalized[:, 2], normalized[:, 1], normalized[:, 0]))
    return readonly_array(normalized[order], dtype=np.uint8)


def _field_errors(
    part: np.ndarray,
    material: np.ndarray,
    emission: np.ndarray,
    guide: np.ndarray,
    legal_tuples: np.ndarray,
) -> list[str]:
    errors: list[str] = []
    expected = (CANVAS_SIZE, CANVAS_SIZE)
    for values, count, name in (
        (part, len(PART_OWNER_NAMES), "part_owner"),
        (material, len(MATERIAL_NAMES), "material"),
        (emission, len(EMISSION_LEVEL_NAMES), "emission_level"),
    ):
        if not isinstance(values, np.ndarray):
            errors.append(f"{name} must be a numpy array")
            continue
        if values.shape != expected:
            errors.append(f"{name} must have shape {expected}")
        if values.dtype != np.uint8:
            errors.append(f"{name} must have dtype uint8")
        if values.size and values.dtype.kind in "ui" and int(values.max()) >= count:
            errors.append(f"{name} contains an out-of-vocabulary value")
    if not isinstance(guide, np.ndarray):
        errors.append("guide must be a numpy array")
    else:
        expected_guide = (len(GUIDE_CHANNEL_NAMES), CANVAS_SIZE, CANVAS_SIZE)
        if guide.shape != expected_guide:
            errors.append(f"guide must have shape {expected_guide}")
        if guide.dtype != np.float32:
            errors.append("guide must have dtype float32")
        if not np.isfinite(guide).all():
            errors.append("guide contains non-finite values")
        elif bool((guide < 0.0).any() or (guide > 1.0).any()):
            errors.append("guide values must stay in [0, 1]")
    if errors:
        return errors

    background = part == 0
    misaligned_background = background & ((material != 0) | (emission != 0))
    if misaligned_background.any():
        errors.append(
            f"{int(misaligned_background.sum())} background pixels carry material or emission"
        )
    if not bool((part != 0).any()):
        errors.append("part_owner contains no foreground")
    unsafe = np.zeros(expected, dtype=bool)
    unsafe[:SAFETY_MARGIN] = True
    unsafe[-SAFETY_MARGIN:] = True
    unsafe[:, :SAFETY_MARGIN] = True
    unsafe[:, -SAFETY_MARGIN:] = True
    unsafe_pixels = int(((part != 0) & unsafe).sum())
    if unsafe_pixels:
        errors.append(
            f"{unsafe_pixels} foreground pixels violate the {SAFETY_MARGIN}-pixel margin"
        )

    observed = np.stack((part, material, emission), axis=-1).reshape(-1, 3)
    legal = {tuple(map(int, row)) for row in legal_tuples}
    invalid = sorted({tuple(map(int, row)) for row in observed} - legal)
    if invalid:
        preview = invalid[:8]
        suffix = f" (+{len(invalid) - 8} more)" if len(invalid) > 8 else ""
        errors.append(f"illegal aligned field tuples: {preview}{suffix}")
    for owner_id, label in ((1, "body"), (3, "head"), (10, "core")):
        if not bool((part == owner_id).any()):
            errors.append(f"required {label} owner is absent")
    return errors


def _components(mask: np.ndarray) -> list[np.ndarray]:
    active = np.asarray(mask, dtype=bool)
    seen = np.zeros_like(active)
    result: list[np.ndarray] = []
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
            result.append(component)
    result.sort(
        key=lambda component: (
            -int(component.sum()),
            int(np.argwhere(component)[0, 0]),
            int(np.argwhere(component)[0, 1]),
        )
    )
    return result


def _topology_errors(part: np.ndarray, family_name: str) -> list[str]:
    # Aura pixels are explicitly non-physical. Every other owner still belongs
    # to a visible rig component even when it is a joint/terminal marker.
    physical = (part != 0) & (part != PART_OWNER_NAMES.index("aura"))
    components = _components(physical)
    if not components:
        return ["physical rig has no pixels"]
    if family_name != "anomaly" and len(components) != 1:
        return [
            f"{family_name} physical rig has {len(components)} components; exactly one is required"
        ]
    if family_name == "anomaly":
        if len(components) > 3:
            return [f"anomaly physical rig has {len(components)} components; maximum is 3"]
        dominant = int(components[0].sum()) / max(1, int(physical.sum()))
        if dominant < 0.85:
            return [f"anomaly dominant component fraction {dominant:.4f} is below 0.85"]
    return []


def _point_errors(
    anatomy: DerivedAnatomy, part: np.ndarray, guide: np.ndarray
) -> list[str]:
    errors: list[str] = []
    if (
        not isinstance(anatomy.source, str)
        or not anatomy.source
        or len(anatomy.source) > 128
    ):
        errors.append("anatomy source must be a nonempty string of at most 128 characters")
    elif anatomy.source not in _ANATOMY_SOURCES:
        errors.append(f"unsupported anatomy source {anatomy.source!r}")
    if not isinstance(anatomy.joints, Mapping):
        errors.append("anatomy joints must be a mapping")
    elif set(anatomy.joints) != set(JOINT_DRIVER):
        errors.append(f"joint keys must be {sorted(JOINT_DRIVER)}")
    if not isinstance(anatomy.sockets, Mapping):
        errors.append("anatomy sockets must be a mapping")
    elif set(anatomy.sockets) != set(SOCKET_DRIVER):
        errors.append(f"socket keys must be {sorted(SOCKET_DRIVER)}")
    for kind, values, channel in (
        ("joint", anatomy.joints, 3),
        ("socket", anatomy.sockets, 4),
    ):
        if not isinstance(values, Mapping):
            continue
        for name, point in values.items():
            if (
                not isinstance(point, tuple)
                or len(point) != 2
                or any(isinstance(value, bool) or not isinstance(value, int) for value in point)
            ):
                errors.append(f"{kind}.{name} must be an integer (x, y) point")
                continue
            x, y = point
            if not (0 <= x < CANVAS_SIZE and 0 <= y < CANVAS_SIZE):
                errors.append(f"{kind}.{name} lies outside the canvas")
                continue
            if part[y, x] == 0:
                errors.append(f"{kind}.{name} lands on neural background")
            # The guide is thickened, so a derived point may sit one pixel from
            # a retained scaffold cell. Reject only complete local disagreement.
            y0, y1 = max(0, y - 1), min(CANVAS_SIZE, y + 2)
            x0, x1 = max(0, x - 1), min(CANVAS_SIZE, x + 2)
            if not bool((guide[channel, y0:y1, x0:x1] > 0.0).any()):
                errors.append(f"{kind}.{name} is unsupported by the conditioning guide")
    if anatomy.source_sha256 is not None and (
        not isinstance(anatomy.source_sha256, str)
        or not _SHA256.fullmatch(anatomy.source_sha256)
    ):
        errors.append("anatomy source_sha256 is not a lowercase SHA-256")
    elif anatomy.source_sha256 is None:
        errors.append("anatomy source_sha256 is required")
    return errors


def _choose_point(
    mask: np.ndarray,
    *,
    guide: np.ndarray | None = None,
    reference: tuple[int, int] | None = None,
    farthest: bool = False,
) -> tuple[int, int]:
    points = np.argwhere(mask)
    if not len(points):
        raise BindingRejected(["cannot derive an anchor from an empty owner mask"])
    if guide is not None:
        scores = guide[points[:, 0], points[:, 1]]
        positive = scores > 0.0
        if positive.any():
            points = points[positive]
            scores = scores[positive]
            points = points[scores == scores.max()]
    if reference is None:
        center_y, center_x = points.mean(axis=0)
        distances = (points[:, 0] - center_y) ** 2 + (points[:, 1] - center_x) ** 2
    else:
        rx, ry = reference
        distances = (points[:, 0] - ry) ** 2 + (points[:, 1] - rx) ** 2
    target = distances.max() if farthest else distances.min()
    candidates = points[distances == target]
    order = np.lexsort((candidates[:, 1], candidates[:, 0]))
    y, x = map(int, candidates[order[0]])
    return x, y


def _automatic_anatomy(part: np.ndarray, guide: np.ndarray) -> DerivedAnatomy:
    missing = [
        PART_OWNER_NAMES[owner]
        for owner in sorted(set(_JOINT_OWNER.values()) | set(_SOCKET_OWNER.values()))
        if not bool((part == owner).any())
    ]
    if missing:
        raise BindingRejected(
            [
                "derived anatomy is required when owner regions are role-overwritten; "
                f"missing direct regions: {missing}"
            ]
        )
    body = part == 1
    root = _choose_point(body, guide=guide[3])
    joints: dict[str, tuple[int, int]] = {"root": root}
    for name, owner in _JOINT_OWNER.items():
        if name == "root":
            continue
        joints[name] = _choose_point(part == owner, guide=guide[3], reference=root)
    sockets: dict[str, tuple[int, int]] = {
        "focus": _choose_point(part == 10, guide=guide[4], reference=root)
    }
    joint_for_socket = {
        "muzzle": "weapon_mount",
        "left_hand": "left_shoulder",
        "right_hand": "right_shoulder",
        "left_foot": "left_hip",
        "right_foot": "right_hip",
        "appendage_tip": "appendage_base",
    }
    for name, owner in _SOCKET_OWNER.items():
        if name == "focus":
            continue
        sockets[name] = _choose_point(
            part == owner,
            guide=guide[4],
            reference=joints[joint_for_socket[name]],
            farthest=True,
        )
    metadata = {"joints": joints, "sockets": sockets, "policy": "guide-owner-v1"}
    return DerivedAnatomy.from_mappings(
        joints,
        sockets,
        source="guide_owner_geometry-v1",
        source_sha256=canonical_json_hash(metadata),
    )


def _anatomy_provenance_errors(
    anatomy: DerivedAnatomy,
    part: np.ndarray,
    guide: np.ndarray,
    *,
    corpus_seed: int | None,
    family_id: int,
    subtype_id: int,
    role_id: int,
) -> list[str]:
    if anatomy.source not in _ANATOMY_SOURCES:
        return []
    try:
        if anatomy.source == "conditioned_anatomy_points_v1":
            if corpus_seed is None:
                return ["conditioned anatomy requires corpus_seed provenance"]
            expected = derive_conditioned_anatomy(
                corpus_seed,
                family=family_id,
                subtype_id=subtype_id,
                role_id=role_id,
            )
        else:
            expected = _automatic_anatomy(part, guide)
    except (BindingRejected, ValueError, TypeError) as error:
        return [f"anatomy provenance could not be reproduced: {error}"]
    errors: list[str] = []
    if dict(anatomy.joints) != dict(expected.joints):
        errors.append("anatomy joints disagree with their declared deterministic source")
    if dict(anatomy.sockets) != dict(expected.sockets):
        errors.append("anatomy sockets disagree with their declared deterministic source")
    if anatomy.source_sha256 != expected.source_sha256:
        errors.append("anatomy source_sha256 disagrees with its deterministic source")
    return errors


def _line_points(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    x0, y0 = start
    x1, y1 = end
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    error = dx + dy
    result: list[tuple[int, int]] = []
    while True:
        result.append((x0, y0))
        if x0 == x1 and y0 == y1:
            return result
        doubled = 2 * error
        if doubled >= dy:
            error += dy
            x0 += sx
        if doubled <= dx:
            error += dx
            y0 += sy


def _nearest_foreground(
    foreground: np.ndarray, point: tuple[int, int], *, radius: int = 2
) -> tuple[int, int] | None:
    x, y = point
    y0, y1 = max(0, y - radius), min(CANVAS_SIZE, y + radius + 1)
    x0, x1 = max(0, x - radius), min(CANVAS_SIZE, x + radius + 1)
    points = np.argwhere(foreground[y0:y1, x0:x1])
    if not len(points):
        return None
    points[:, 0] += y0
    points[:, 1] += x0
    distance = (points[:, 0] - y) ** 2 + (points[:, 1] - x) ** 2
    candidates = points[distance == distance.min()]
    order = np.lexsort((candidates[:, 1], candidates[:, 0]))
    py, px = map(int, candidates[order[0]])
    return px, py


def _anatomy_point(
    anatomy: DerivedAnatomy, descriptor: tuple[str, str]
) -> tuple[int, int]:
    kind, name = descriptor
    return anatomy.joints[name] if kind == "joint" else anatomy.sockets[name]


def _build_driver_index(
    part: np.ndarray, anatomy: DerivedAnatomy
) -> np.ndarray:
    foreground = part != 0
    result = np.full(part.shape, BACKGROUND_DRIVER, dtype=np.uint8)
    for owner_id, driver in _DIRECT_OWNER_DRIVER.items():
        result[part == owner_id] = DRIVER_INDEX[driver]

    # Exact derived anchors are authoritative binding metadata, not pixel
    # replacements. Assign their existing neural tuples to their named driver.
    protected_anchors = np.zeros(part.shape, dtype=bool)
    for values, drivers in (
        (anatomy.joints, JOINT_DRIVER),
        (anatomy.sockets, SOCKET_DRIVER),
    ):
        for name, point in values.items():
            x, y = point
            result[y, x] = DRIVER_INDEX[drivers[name]]
            protected_anchors[y, x] = True

    # Role conditioning can replace all direct owner tokens for a limb. Seed
    # each missing driver from its derived bone, using only foreground pixels.
    flexible = part >= 11
    for driver in DRIVER_NAMES:
        driver_id = DRIVER_INDEX[driver]
        first, second = _DRIVER_SEGMENT[driver]
        for point in _line_points(
            _anatomy_point(anatomy, first), _anatomy_point(anatomy, second)
        ):
            snapped = _nearest_foreground(foreground, point)
            if snapped is None:
                continue
            x, y = snapped
            if protected_anchors[y, x] and result[y, x] != driver_id:
                continue
            if result[y, x] == BACKGROUND_DRIVER or flexible[y, x]:
                result[y, x] = driver_id

    # Assign every remaining non-direct owner pixel to the nearest seeded
    # driver. Ties resolve in DRIVER_NAMES order and are therefore replayable.
    unassigned = foreground & (result == BACKGROUND_DRIVER)
    if unassigned.any():
        seed_points: list[np.ndarray] = []
        for driver_id in range(len(DRIVER_NAMES)):
            points = np.argwhere(result == driver_id)
            if not len(points):
                raise BindingRejected(
                    [f"driver {DRIVER_NAMES[driver_id]} has no neural pixel seed"]
                )
            seed_points.append(points)
        for y, x in np.argwhere(unassigned):
            best_driver = 0
            best_distance = math.inf
            for driver_id, points in enumerate(seed_points):
                distance = int(
                    np.min((points[:, 0] - y) ** 2 + (points[:, 1] - x) ** 2)
                )
                if distance < best_distance:
                    best_distance = distance
                    best_driver = driver_id
            result[y, x] = best_driver

    errors: list[str] = []
    for driver_id, driver in enumerate(DRIVER_NAMES):
        pixel_count = int((result == driver_id).sum())
        if pixel_count < MIN_DRIVER_PIXELS[driver]:
            errors.append(
                f"driver {driver} owns {pixel_count} neural pixels; "
                f"minimum is {MIN_DRIVER_PIXELS[driver]}"
            )
    if errors:
        raise BindingRejected(errors)
    return readonly_array(result, dtype=np.uint8)


def _anchors(
    part: np.ndarray,
    anatomy: DerivedAnatomy,
    driver_index: np.ndarray,
) -> tuple[Mapping[str, RigAnchor], Mapping[str, RigAnchor]]:
    def support(point: tuple[int, int], driver: str) -> tuple[int, int]:
        points = np.argwhere(driver_index == DRIVER_INDEX[driver])
        x, y = point
        distance = (points[:, 0] - y) ** 2 + (points[:, 1] - x) ** 2
        candidates = points[distance == distance.min()]
        order = np.lexsort((candidates[:, 1], candidates[:, 0]))
        sy, sx = map(int, candidates[order[0]])
        # Several valid grammars co-locate root, focus, and appendage pivots.
        # A flattened owner field can assign that one tuple to only one driver;
        # the child therefore binds to its nearest bone-supported pixel.
        if max(abs(sx - x), abs(sy - y)) > 6:
            raise BindingRejected(
                [f"anchor at {point} has no {driver} support pixel within six cells"]
            )
        return sx, sy

    joints = {
        name: RigAnchor(
            name=name,
            kind="joint",
            point=point,
            support_point=support(point, JOINT_DRIVER[name]),
            driver=JOINT_DRIVER[name],
            observed_owner=int(part[point[1], point[0]]),
            source=anatomy.source,
        )
        for name, point in anatomy.joints.items()
    }
    sockets = {
        name: RigAnchor(
            name=name,
            kind="socket",
            point=point,
            support_point=support(point, SOCKET_DRIVER[name]),
            driver=SOCKET_DRIVER[name],
            observed_owner=int(part[point[1], point[0]]),
            source=anatomy.source,
        )
        for name, point in anatomy.sockets.items()
    }
    return MappingProxyType(joints), MappingProxyType(sockets)


def _component_metadata(
    part: np.ndarray, driver_index: np.ndarray, family_name: str
) -> list[dict[str, Any]]:
    physical = (part != 0) & (part != PART_OWNER_NAMES.index("aura"))
    components = _components(physical)
    result: list[dict[str, Any]] = []
    for index, component in enumerate(components):
        driver_counts = [int(((driver_index == driver) & component).sum()) for driver in range(8)]
        driver_id = int(np.argmax(driver_counts))
        result.append(
            {
                "component": index,
                "pixel_count": int(component.sum()),
                "dominant_driver": DRIVER_NAMES[driver_id],
                "logical_parent": None if index == 0 else "body",
                "joint_type": (
                    "orbital" if index > 0 and family_name == "anomaly" else "fixed"
                ),
                "logical_only": index > 0,
            }
        )
    return result


def _graph_metadata(
    family_name: str,
    joints: Mapping[str, RigAnchor],
    driver_index: np.ndarray,
) -> dict[str, Any]:
    parent = {
        "body": None,
        "head": "body",
        "left_arm": "body",
        "right_arm": "body",
        "left_leg": "body",
        "right_leg": "body",
        "appendage": "body",
        "weapon": "right_arm" if family_name == "humanoid" else "head",
    }
    joint_for_driver = {
        "body": "root",
        "head": "head",
        "left_arm": "left_shoulder",
        "right_arm": "right_shoulder",
        "left_leg": "left_hip",
        "right_leg": "right_hip",
        "appendage": "appendage_base",
        "weapon": "weapon_mount",
    }
    joint_type = {
        "body": "fixed",
        "head": "hinge",
        "left_arm": "hinge",
        "right_arm": "hinge",
        "left_leg": "hinge",
        "right_leg": "hinge",
        "appendage": "orbital" if family_name == "anomaly" else "spline",
        "weapon": "hinge",
    }
    nodes = []
    edges = []
    for driver_id, name in enumerate(DRIVER_NAMES):
        joint = joint_for_driver[name]
        nodes.append(
            {
                "id": name,
                "parent": parent[name],
                "pivot": list(joints[joint].point),
                "joint": joint,
                "joint_type": joint_type[name],
                "pixel_count": int((driver_index == driver_id).sum()),
            }
        )
        if parent[name] is not None:
            edges.append(
                {
                    "parent": parent[name],
                    "child": name,
                    "joint": joint,
                    "joint_type": joint_type[name],
                }
            )
    return {"root": "body", "nodes": nodes, "edges": edges, "connected": True}


def _owner_layers(
    part: np.ndarray,
    material: np.ndarray,
    emission: np.ndarray,
    driver_index: np.ndarray,
) -> tuple[np.ndarray, tuple[OwnerLayerBinding, ...]]:
    masks = np.stack([part == owner for owner in range(1, len(PART_OWNER_NAMES))]).astype(
        np.uint8
    )
    layers: list[OwnerLayerBinding] = []
    for owner_id, mask in enumerate(masks, start=1):
        driver_ids = sorted(set(map(int, np.unique(driver_index[mask.astype(bool)]))))
        drivers = tuple(
            DRIVER_NAMES[index] for index in driver_ids if index < len(DRIVER_NAMES)
        )
        layers.append(
            OwnerLayerBinding(
                owner_id=owner_id,
                owner_name=PART_OWNER_NAMES[owner_id],
                pixel_count=int(mask.sum()),
                drivers=drivers,
                tuple_sha256=owner_tuple_hash(
                    owner_id, mask.astype(bool), material, emission
                ),
            )
        )
    return readonly_array(masks, dtype=np.uint8), tuple(layers)


def _anatomy_hash(anatomy: DerivedAnatomy) -> str:
    return canonical_json_hash(
        {
            "source": anatomy.source,
            "joints": {name: list(point) for name, point in anatomy.joints.items()},
            "sockets": {name: list(point) for name, point in anatomy.sockets.items()},
        }
    )


def _manifest(
    *,
    sample_id: str,
    family_name: str,
    family_id: int,
    subtype_id: int,
    role_id: int,
    corpus_seed: int | None,
    part: np.ndarray,
    material: np.ndarray,
    emission: np.ndarray,
    guide: np.ndarray,
    genes: np.ndarray | None,
    legal_tuples: np.ndarray,
    anatomy: DerivedAnatomy,
    joints: Mapping[str, RigAnchor],
    sockets: Mapping[str, RigAnchor],
    owner_layers: tuple[OwnerLayerBinding, ...],
    driver_index: np.ndarray,
    upstream_hashes: Mapping[str, str],
) -> dict[str, Any]:
    components = _component_metadata(part, driver_index, family_name)
    physical_pixels = sum(int(item["pixel_count"]) for item in components)
    base: dict[str, Any] = {
        "format": BINDING_FORMAT,
        "id": sample_id,
        "binder_version": BINDER_VERSION,
        "canvas_size": [CANVAS_SIZE, CANVAS_SIZE],
        "safety_margin": SAFETY_MARGIN,
        "condition": {
            "family": family_name,
            "family_id": family_id,
            "subtype_id": subtype_id,
            "subtype_name": SUBTYPE_NAMES[subtype_id],
            "role_id": role_id,
            "role_name": ROLE_NAMES[role_id],
            "corpus_seed": corpus_seed,
        },
        "owner_names": list(PART_OWNER_NAMES),
        "driver_names": list(DRIVER_NAMES),
        "owner_layers": [layer.metadata() for layer in owner_layers],
        "joint_bindings": {name: anchor.metadata() for name, anchor in joints.items()},
        "socket_bindings": {name: anchor.metadata() for name, anchor in sockets.items()},
        "graph": _graph_metadata(family_name, joints, driver_index),
        "topology": {
            "policy": (
                "anomaly-dominant-plus-orbitals-v1"
                if family_name == "anomaly"
                else "single-physical-component-v1"
            ),
            "physical_component_count": len(components),
            "physical_pixels": physical_pixels,
            "dominant_component_fraction": (
                float(components[0]["pixel_count"]) / max(1, physical_pixels)
            ),
            "logical_graph_connected": True,
            "components": components,
        },
        "source": {
            "pixel_authority": "raw_neural_aligned_fields",
            "procedural_pixel_substitution": False,
            "raw_fields_sha256": aligned_fields_hash(part, material, emission),
            "guide_sha256": array_hash("conditioning_guide", guide),
            "genes_sha256": (
                array_hash("condition_genes", genes) if genes is not None else None
            ),
            "legal_tuples_sha256": tuple_fingerprint(legal_tuples),
            "anatomy_sha256": _anatomy_hash(anatomy),
            "anatomy_source": anatomy.source,
            "anatomy_source_sha256": anatomy.source_sha256,
            "driver_index_sha256": array_hash("driver_index", driver_index),
            "binder_source_sha256": binder_source_hash(),
            "upstream_hashes": dict(sorted(upstream_hashes.items())),
        },
        "adapter": {
            "format": ADAPTER_FORMAT,
            "motion_renderer_version": "graph-layer-rig-v1",
            "matrix_convention": "source_to_destination_affine_3x3",
            "tuple_policy": "copy_existing_tuple_only",
            "rest_frame_exact": True,
            "compiler_integration": "driver-matrix-adapter",
        },
    }
    base["hashes"] = {"binding_sha256": canonical_json_hash(base)}
    return base


def bind_neural_fields(
    part_owner: np.ndarray,
    material: np.ndarray,
    emission_level: np.ndarray,
    guide: np.ndarray,
    *,
    family: int | str,
    subtype_id: int,
    role_id: int,
    anatomy: DerivedAnatomy | None = None,
    sample_id: str | None = None,
    legal_tuples: np.ndarray | Iterable[tuple[int, int, int]] | None = None,
    genes: np.ndarray | None = None,
    corpus_seed: int | None = None,
    upstream_hashes: Mapping[str, str] | None = None,
) -> NeuralRigBinding:
    """Bind authoritative neural fields to a logical rig without rewriting them.

    `anatomy` supplies points only. It may come from the conditioning source or
    from scaffold analysis, but it cannot carry a procedural mask into this API.
    """
    family_id = _family_index(family)
    family_name = FAMILIES[family_id]
    errors: list[str] = []
    if isinstance(subtype_id, bool) or not isinstance(subtype_id, (int, np.integer)):
        errors.append("subtype_id must be an integer")
    elif not 0 <= int(subtype_id) < len(SUBTYPE_NAMES):
        errors.append("subtype_id must be in [0, 19]")
    elif int(subtype_id) // 4 != family_id:
        errors.append("subtype_id does not belong to the requested family")
    if isinstance(role_id, bool) or not isinstance(role_id, (int, np.integer)):
        errors.append("role_id must be an integer")
    elif not 0 <= int(role_id) < len(ROLE_NAMES):
        errors.append("role_id must be in [0, 7]")
    if corpus_seed is not None and (
        isinstance(corpus_seed, bool)
        or not isinstance(corpus_seed, (int, np.integer))
        or not 0 <= int(corpus_seed) <= 0xFFFFFFFF
    ):
        errors.append("corpus_seed must be null or an unsigned 32-bit integer")
    if genes is not None:
        if not isinstance(genes, np.ndarray) or genes.shape != (24,) or genes.dtype != np.float32:
            errors.append("genes must be null or a float32 vector of length 24")
        elif not np.isfinite(genes).all():
            errors.append("genes contain non-finite values")
    if errors:
        raise BindingRejected(errors)

    normalized_legal = _normalize_legal_tuples(legal_tuples)
    errors = _field_errors(
        part_owner, material, emission_level, guide, normalized_legal
    )
    errors.extend(_topology_errors(part_owner, family_name) if not errors else [])
    if errors:
        raise BindingRejected(errors)

    part = readonly_array(part_owner, dtype=np.uint8)
    material_values = readonly_array(material, dtype=np.uint8)
    emission_values = readonly_array(emission_level, dtype=np.uint8)
    guide_values = readonly_array(guide, dtype=np.float32)
    genes_values = readonly_array(genes, dtype=np.float32) if genes is not None else None

    if anatomy is not None and not isinstance(anatomy, DerivedAnatomy):
        raise BindingRejected(["anatomy must be a DerivedAnatomy value or null"])
    resolved_anatomy = anatomy or _automatic_anatomy(part, guide_values)
    point_errors = _point_errors(resolved_anatomy, part, guide_values)
    if not point_errors:
        point_errors.extend(
            _anatomy_provenance_errors(
                resolved_anatomy,
                part,
                guide_values,
                corpus_seed=int(corpus_seed) if corpus_seed is not None else None,
                family_id=family_id,
                subtype_id=int(subtype_id),
                role_id=int(role_id),
            )
        )
    if point_errors:
        raise BindingRejected(point_errors)

    if upstream_hashes is not None and not isinstance(upstream_hashes, Mapping):
        raise BindingRejected(["upstream_hashes must be a mapping or null"])
    if upstream_hashes is not None and len(upstream_hashes) > 32:
        raise BindingRejected(["upstream_hashes may contain at most 32 entries"])
    upstream = dict(upstream_hashes or {})
    invalid_hashes = [
        name
        for name, value in upstream.items()
        if not isinstance(name, str)
        or not _UPSTREAM_NAME.fullmatch(name)
        or not isinstance(value, str)
        or not _SHA256.fullmatch(value)
    ]
    if invalid_hashes:
        raise BindingRejected(
            [f"invalid upstream SHA-256 entries: {sorted(map(str, invalid_hashes))}"]
        )
    evaluator_fingerprint = upstream.get("legal_tuple_fingerprint")
    if (
        evaluator_fingerprint is not None
        and evaluator_fingerprint != evaluator_tuple_fingerprint(normalized_legal)
    ):
        raise BindingRejected(
            ["upstream legal_tuple_fingerprint disagrees with legal_tuples"]
        )

    driver_index = _build_driver_index(part, resolved_anatomy)
    joints, sockets = _anchors(part, resolved_anatomy, driver_index)
    owner_masks, owner_layers = _owner_layers(
        part, material_values, emission_values, driver_index
    )
    raw_hash = aligned_fields_hash(part, material_values, emission_values)
    resolved_id = sample_id or f"{family_name}_{raw_hash[:16]}"
    if not isinstance(resolved_id, str) or not _SAMPLE_ID.fullmatch(resolved_id):
        raise BindingRejected(
            ["sample_id must be 1-128 safe alphanumeric/dot/underscore/hyphen characters"]
        )

    manifest = _manifest(
        sample_id=resolved_id,
        family_name=family_name,
        family_id=family_id,
        subtype_id=int(subtype_id),
        role_id=int(role_id),
        corpus_seed=int(corpus_seed) if corpus_seed is not None else None,
        part=part,
        material=material_values,
        emission=emission_values,
        guide=guide_values,
        genes=genes_values,
        legal_tuples=normalized_legal,
        anatomy=resolved_anatomy,
        joints=joints,
        sockets=sockets,
        owner_layers=owner_layers,
        driver_index=driver_index,
        upstream_hashes=upstream,
    )
    return NeuralRigBinding(
        sample_id=resolved_id,
        family=family_name,
        family_id=family_id,
        subtype_id=int(subtype_id),
        role_id=int(role_id),
        corpus_seed=int(corpus_seed) if corpus_seed is not None else None,
        part_owner=part,
        material=material_values,
        emission_level=emission_values,
        guide=guide_values,
        genes=genes_values,
        legal_tuples=normalized_legal,
        owner_masks=owner_masks,
        driver_index=driver_index,
        joints=joints,
        sockets=sockets,
        owner_layers=owner_layers,
        anatomy=resolved_anatomy,
        upstream_hashes=MappingProxyType(upstream),
        manifest=manifest,
    )


def derive_conditioned_anatomy(
    corpus_seed: int,
    *,
    family: int | str,
    subtype_id: int,
    role_id: int,
) -> DerivedAnatomy:
    """Rebuild only named conditioning points from a versioned corpus seed.

    The temporary procedural specimen is discarded. No layer, token, color, or
    tuple from it enters a neural binding.
    """
    family_id = _family_index(family)
    if (
        isinstance(corpus_seed, (bool, np.bool_))
        or not isinstance(corpus_seed, (int, np.integer))
        or not 0 <= int(corpus_seed) <= 0xFFFFFFFF
    ):
        raise BindingRejected(["corpus_seed must be an unsigned 32-bit integer"])
    if (
        isinstance(subtype_id, (bool, np.bool_))
        or not isinstance(subtype_id, (int, np.integer))
        or not 0 <= int(subtype_id) < len(SUBTYPE_NAMES)
        or int(subtype_id) // 4 != family_id
    ):
        raise BindingRejected(["subtype_id does not belong to the requested family"])
    if (
        isinstance(role_id, (bool, np.bool_))
        or not isinstance(role_id, (int, np.integer))
        or not 0 <= int(role_id) < len(ROLE_NAMES)
    ):
        raise BindingRejected(["role_id must be in [0, 7]"])
    from ..morphology.genome import genome_from_seed
    from ..morphology.render import render_specimen

    genome = replace(
        genome_from_seed(int(corpus_seed), family_id),
        silhouette_variant=int(subtype_id) % 4,
        subtype_id=int(subtype_id),
        role_id=int(role_id),
    )
    specimen = render_specimen(genome)
    payload = {
        "policy": "conditioned-anatomy-points-v1",
        "corpus_seed": int(corpus_seed),
        "family_id": family_id,
        "subtype_id": int(subtype_id),
        "role_id": int(role_id),
        "joints": specimen.joints,
        "sockets": specimen.sockets,
    }
    return DerivedAnatomy.from_mappings(
        specimen.joints,
        specimen.sockets,
        source="conditioned_anatomy_points_v1",
        source_sha256=canonical_json_hash(payload),
    )


_RAW_ARRAY_SPECS: dict[str, tuple[tuple[int, ...], np.dtype[Any]]] = {
    "part": ((CANVAS_SIZE, CANVAS_SIZE), np.dtype(np.uint8)),
    "material": ((CANVAS_SIZE, CANVAS_SIZE), np.dtype(np.uint8)),
    "emission": ((CANVAS_SIZE, CANVAS_SIZE), np.dtype(np.uint8)),
    "guide": (
        (len(GUIDE_CHANNEL_NAMES), CANVAS_SIZE, CANVAS_SIZE),
        np.dtype(np.float32),
    ),
    "genes": ((24,), np.dtype(np.float32)),
    "target_part": ((CANVAS_SIZE, CANVAS_SIZE), np.dtype(np.uint8)),
    "target_material": ((CANVAS_SIZE, CANVAS_SIZE), np.dtype(np.uint8)),
    "target_emission": ((CANVAS_SIZE, CANVAS_SIZE), np.dtype(np.uint8)),
    "morphology": ((1,), np.dtype(np.uint8)),
    "subtype": ((1,), np.dtype(np.uint8)),
    "role": ((1,), np.dtype(np.uint8)),
    "source_index": ((1,), np.dtype(np.int64)),
    "corpus_seed": ((1,), np.dtype(np.uint32)),
    "sample_seed": ((1,), np.dtype(np.uint64)),
}
_RAW_ARCHIVE_KEYS = frozenset({"format", *_RAW_ARRAY_SPECS})


def _read_bounded(path: Path, *, limit: int, label: str) -> bytes:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as handle:
        payload = handle.read(limit + 1)
    if len(payload) > limit:
        raise BindingRejected([f"{label} exceeds the {limit}-byte admission bound"])
    if not payload:
        raise BindingRejected([f"{label} is empty"])
    return payload


def _strict_json(payload: bytes) -> dict[str, Any]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number {value}")

    try:
        decoded = payload.decode("utf-8")
        result = json.loads(
            decoded,
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as error:
        raise BindingRejected([f"raw manifest is not strict UTF-8 JSON: {error}"]) from error
    if not isinstance(result, dict):
        raise BindingRejected(["raw manifest root must be an object"])
    return result


def _validate_raw_manifest_schema(payload: Mapping[str, Any]) -> None:
    project = Path(__file__).resolve().parents[2]
    schema_path = project / "shared" / "schema" / "multifield_raw_sample.schema.json"
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        errors = sorted(
            Draft202012Validator(schema).iter_errors(dict(payload)),
            key=lambda error: tuple(map(str, error.absolute_path)),
        )
    except (OSError, json.JSONDecodeError, RecursionError) as error:
        raise BindingRejected([f"raw manifest schema could not be loaded: {error}"]) from error
    if errors:
        messages = [
            f"{'/'.join(map(str, error.absolute_path)) or '<root>'}: {error.message}"
            for error in errors[:16]
        ]
        if len(errors) > 16:
            messages.append(f"raw manifest has {len(errors) - 16} additional schema errors")
        raise BindingRejected([f"raw manifest schema: {message}" for message in messages])


def _preflight_raw_npz(payload: bytes) -> None:
    try:
        with zipfile.ZipFile(io.BytesIO(payload), mode="r") as archive:
            infos = archive.infolist()
            if len(infos) != len(_RAW_ARCHIVE_KEYS):
                raise BindingRejected(
                    [
                        "raw archive member count is not exact: "
                        f"expected {len(_RAW_ARCHIVE_KEYS)}, observed {len(infos)}"
                    ]
                )
            expected_members = {f"{name}.npy" for name in _RAW_ARCHIVE_KEYS}
            observed_members = [info.filename for info in infos]
            if len(set(observed_members)) != len(observed_members):
                raise BindingRejected(["raw archive contains duplicate ZIP members"])
            if set(observed_members) != expected_members:
                missing = sorted(expected_members - set(observed_members))
                extra = sorted(set(observed_members) - expected_members)
                raise BindingRejected(
                    [f"raw archive member set is not exact; missing={missing}, extra={extra}"]
                )
            total_uncompressed = 0
            for info in infos:
                if info.is_dir() or info.flag_bits & 0x1:
                    raise BindingRejected(
                        [f"raw archive member {info.filename!r} is a directory or encrypted"]
                    )
                total_uncompressed += int(info.file_size)
                if total_uncompressed > _RAW_ARCHIVE_MAX_UNCOMPRESSED_BYTES:
                    raise BindingRejected(
                        [
                            "raw archive exceeds the "
                            f"{_RAW_ARCHIVE_MAX_UNCOMPRESSED_BYTES}-byte uncompressed bound"
                        ]
                    )
                key = info.filename[:-4]
                with archive.open(info, mode="r") as member:
                    version = np.lib.format.read_magic(member)
                    if version == (1, 0):
                        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(
                            member
                        )
                    elif version == (2, 0):
                        shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(
                            member
                        )
                    else:
                        raise BindingRejected(
                            [
                                f"raw archive member {info.filename!r} uses unsupported "
                                f"NPY header version {version}"
                            ]
                        )
                if fortran_order:
                    raise BindingRejected(
                        [f"raw archive member {info.filename!r} must use C array order"]
                    )
                dtype = np.dtype(dtype)
                if dtype.hasobject:
                    raise BindingRejected(
                        [f"raw archive member {info.filename!r} has an object dtype"]
                    )
                if key == "format":
                    if shape != (1,) or dtype.kind != "U" or dtype.itemsize > 512:
                        raise BindingRejected(
                            ["raw archive format must be one bounded Unicode scalar"]
                        )
                    continue
                expected_shape, expected_dtype = _RAW_ARRAY_SPECS[key]
                if tuple(shape) != expected_shape:
                    raise BindingRejected(
                        [
                            f"raw archive {key} shape is {tuple(shape)}, "
                            f"expected {expected_shape}"
                        ]
                    )
                if dtype != expected_dtype or (dtype.itemsize > 1 and not dtype.isnative):
                    raise BindingRejected(
                        [
                            f"raw archive {key} dtype is {dtype.str}, "
                            f"expected native {expected_dtype.str}"
                        ]
                    )
    except BindingRejected:
        raise
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise BindingRejected([f"raw archive is not a valid bounded NPZ: {error}"]) from error


def _safe_artifact_path(run_root: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise BindingRejected(["raw manifest artifact path must be a POSIX relative path"])
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise BindingRejected([f"raw manifest artifact path is unsafe: {relative!r}"])
    candidate = (run_root / Path(*pure.parts)).resolve()
    if not candidate.is_relative_to(run_root):
        raise BindingRejected([f"raw manifest artifact path escapes its run: {relative!r}"])
    return candidate


def _condition_errors(
    condition: Any,
    *,
    sample_id: str,
    family_id: int,
    subtype_id: int,
    role_id: int,
    source_index: int,
    sample_seed: int,
) -> list[str]:
    if not isinstance(condition, Mapping):
        return ["raw manifest condition is not an object"]
    required = {
        "ordinal",
        "sample_id",
        "grid_mode",
        "source_index",
        "variation",
        "sample_seed",
        "morphology_id",
        "morphology_name",
        "subtype_id",
        "subtype_name",
        "role_id",
        "role_name",
    }
    if set(condition) != required:
        return [
            "raw manifest condition keys are not exact: "
            f"missing={sorted(required - set(condition))}, "
            f"extra={sorted(set(condition) - required)}"
        ]
    errors: list[str] = []
    expected = {
        "sample_id": sample_id,
        "source_index": source_index,
        "sample_seed": sample_seed,
        "morphology_id": family_id,
        "morphology_name": FAMILIES[family_id],
        "subtype_id": subtype_id,
        "subtype_name": SUBTYPE_NAMES[subtype_id],
        "role_id": role_id,
        "role_name": ROLE_NAMES[role_id],
    }
    for name, value in expected.items():
        if condition.get(name) != value or (
            isinstance(value, int) and type(condition.get(name)) is not int
        ):
            errors.append(f"raw manifest condition.{name} disagrees with the archive")
    for name in ("ordinal", "variation"):
        value = condition.get(name)
        if type(value) is not int or value < 0:
            errors.append(f"raw manifest condition.{name} must be a nonnegative integer")
    if condition.get("grid_mode") not in {"fixed", "stratified", "exhaustive"}:
        errors.append("raw manifest condition.grid_mode is unsupported")
    return errors


def bind_raw_sample_archive(
    archive_path: Path,
    *,
    raw_manifest_path: Path | None = None,
    anatomy: DerivedAnatomy | None = None,
    legal_tuples: np.ndarray | Iterable[tuple[int, int, int]] | None = None,
) -> NeuralRigBinding:
    """Bind one accepted evaluator raw NPZ/manifest pair on CPU."""
    path = Path(archive_path).resolve()
    archive_bytes = _read_bounded(
        path,
        limit=_RAW_ARCHIVE_MAX_BYTES,
        label="raw archive",
    )
    _preflight_raw_npz(archive_bytes)
    try:
        with np.load(io.BytesIO(archive_bytes), allow_pickle=False) as archive:
            if set(archive.files) != _RAW_ARCHIVE_KEYS:
                raise BindingRejected(["raw archive key set changed after preflight"])
            stored_format_values = np.asarray(archive["format"])
            stored_format = str(stored_format_values.item())
            if stored_format != _RAW_SAMPLE_FORMAT:
                raise BindingRejected([f"unsupported raw archive format {stored_format!r}"])
            arrays = {
                name: np.asarray(archive[name]).copy()
                for name in _RAW_ARRAY_SPECS
            }
    except BindingRejected:
        raise
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise BindingRejected([f"raw archive could not be loaded after preflight: {error}"]) from error

    family_id = int(arrays["morphology"].item())
    subtype_id = int(arrays["subtype"].item())
    role_id = int(arrays["role"].item())
    source_index = int(arrays["source_index"].item())
    corpus_seed = int(arrays["corpus_seed"].item())
    sample_seed = int(arrays["sample_seed"].item())
    normalized_legal = _normalize_legal_tuples(legal_tuples)
    resolved_anatomy = anatomy or derive_conditioned_anatomy(
        corpus_seed,
        family=family_id,
        subtype_id=subtype_id,
        role_id=role_id,
    )
    if raw_manifest_path is None:
        raise BindingRejected(
            ["raw_manifest_path is required to prove evaluator acceptance and provenance"]
        )
    manifest_path = Path(raw_manifest_path).resolve()
    manifest_bytes = _read_bounded(
        manifest_path,
        limit=_RAW_MANIFEST_MAX_BYTES,
        label="raw manifest",
    )
    payload = _strict_json(manifest_bytes)
    _validate_raw_manifest_schema(payload)

    archive_sha256 = hashlib.sha256(archive_bytes).hexdigest()
    expected_raw = aligned_fields_hash(
        arrays["part"], arrays["material"], arrays["emission"]
    )
    if payload.get("format") != _RAW_SAMPLE_FORMAT:
        raise BindingRejected(["raw manifest format is unsupported"])
    if payload.get("raw_fields_sha256") != expected_raw:
        raise BindingRejected(["raw manifest field hash disagrees with the archive"])
    validation = payload.get("validation")
    if not isinstance(validation, Mapping):
        raise BindingRejected(["raw manifest validation is not an object"])
    if validation.get("format") != _RAW_VALIDATION_FORMAT:
        raise BindingRejected(["raw manifest embedded validation format is unsupported"])
    if validation.get("raw_fields_sha256") != expected_raw:
        raise BindingRejected(["embedded validation field hash disagrees with the archive"])
    if validation.get("hard_valid") is not True or validation.get("accepted") is not True:
        raise BindingRejected(["raw sample was not accepted and hard-valid"])
    hard_gates = validation.get("hard_gates")
    if (
        validation.get("errors") != []
        or not isinstance(hard_gates, Mapping)
        or not hard_gates
        or any(value is not True for value in hard_gates.values())
    ):
        raise BindingRejected(["raw sample has validation errors or a failed hard gate"])

    sample_id = path.stem
    if manifest_path.stem != sample_id:
        raise BindingRejected(["raw archive and manifest filenames identify different samples"])
    condition_errors = _condition_errors(
        payload.get("condition"),
        sample_id=sample_id,
        family_id=family_id,
        subtype_id=subtype_id,
        role_id=role_id,
        source_index=source_index,
        sample_seed=sample_seed,
    )
    if condition_errors:
        raise BindingRejected(condition_errors)
    if validation.get("sample_id") != sample_id:
        raise BindingRejected(["embedded validation sample_id disagrees with the archive"])

    run_root = manifest_path.parent.parent.resolve()
    artifacts = payload["artifacts"]
    for artifact_name in ("fields", "rgba", "emission"):
        _safe_artifact_path(run_root, artifacts[artifact_name]["path"])
    fields_artifact = artifacts["fields"]
    if _safe_artifact_path(run_root, fields_artifact["path"]) != path:
        raise BindingRejected(["raw manifest fields path does not resolve to the passed archive"])
    if fields_artifact.get("sha256") != archive_sha256:
        raise BindingRejected(["raw manifest archive SHA-256 is incorrect"])
    if type(fields_artifact.get("bytes")) is not int or fields_artifact["bytes"] != len(
        archive_bytes
    ):
        raise BindingRejected(["raw manifest archive byte count is incorrect"])

    evaluator_fingerprint = evaluator_tuple_fingerprint(normalized_legal)
    if payload.get("legal_tuple_fingerprint") != evaluator_fingerprint:
        raise BindingRejected(
            ["raw manifest legal tuple fingerprint disagrees with legal_tuples"]
        )
    tuple_validation = validation.get("tuples")
    if (
        not isinstance(tuple_validation, Mapping)
        or type(tuple_validation.get("legal_tuple_count")) is not int
        or tuple_validation.get("legal_tuple_count") != len(normalized_legal)
    ):
        raise BindingRejected(
            ["embedded validation legal tuple count disagrees with legal_tuples"]
        )

    upstream: dict[str, str] = {
        "raw_archive_sha256": archive_sha256,
        "raw_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    }
    for key in (
        "checkpoint_sha256",
        "canonical_ema_hash",
        "corpus_sha256",
        "training_source_hash",
        "evaluation_source_hash",
        "legal_tuple_fingerprint",
    ):
        upstream[key] = str(payload[key])

    return bind_neural_fields(
        arrays["part"],
        arrays["material"],
        arrays["emission"],
        arrays["guide"],
        family=family_id,
        subtype_id=subtype_id,
        role_id=role_id,
        anatomy=resolved_anatomy,
        sample_id=sample_id,
        legal_tuples=normalized_legal,
        genes=arrays["genes"],
        corpus_seed=corpus_seed,
        upstream_hashes=upstream,
    )
