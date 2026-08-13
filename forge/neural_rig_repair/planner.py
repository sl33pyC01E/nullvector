from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..morphology.constants import FAMILIES, PART_OWNER_NAMES
from ..neural_rig_bridge import BindingRejected, bind_raw_sample_archive
from ..neural_rig_bridge.binding import derive_conditioned_anatomy
from ..neural_rig_bridge.hashing import binder_source_hash
from ..neural_rig_bridge.model import (
    BACKGROUND_DRIVER,
    DRIVER_INDEX,
    DRIVER_NAMES,
    JOINT_DRIVER,
    MIN_DRIVER_PIXELS,
    SOCKET_DRIVER,
)
from .constants import (
    AURA_OWNER_ID,
    EXPECTED_BRIDGE_SOURCE_SHA256,
    MAX_ANCHOR_DISPLACEMENT,
    MAX_PLAN_BYTES,
    PLAN_SCHEMA,
    PLAN_FORMAT,
    PROJECT_ROOT,
    REPAIR_VERSION,
    REPAIR_MIN_DRIVER_PIXELS,
    REQUIRED_OWNER_IDS,
)
from .hashing import array_sha256, canonical_json_bytes, sha256_bytes
from .model import RepairAnchor, RepairSource, RepairSourceSample, readonly_array
from .schema import load_schema_json


DIRECT_OWNER_DRIVER = {
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
DRIVER_SEGMENTS = {
    "body": (("joint", "root"), ("socket", "focus")),
    "head": (("joint", "head"), ("joint", "head")),
    "left_arm": (("joint", "left_shoulder"), ("socket", "left_hand")),
    "right_arm": (("joint", "right_shoulder"), ("socket", "right_hand")),
    "left_leg": (("joint", "left_hip"), ("socket", "left_foot")),
    "right_leg": (("joint", "right_hip"), ("socket", "right_foot")),
    "appendage": (("joint", "appendage_base"), ("socket", "appendage_tip")),
    "weapon": (("joint", "weapon_mount"), ("socket", "muzzle")),
}


def _relative(path: Path) -> str:
    return Path(path).resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def _components(mask: np.ndarray) -> list[np.ndarray]:
    active = np.asarray(mask, dtype=bool)
    seen = np.zeros_like(active)
    result: list[np.ndarray] = []
    for start_y, start_x in np.argwhere(active):
        if seen[start_y, start_x]:
            continue
        component = np.zeros_like(active)
        stack = [(int(start_y), int(start_x))]
        seen[start_y, start_x] = True
        while stack:
            y, x = stack.pop()
            component[y, x] = True
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if (
                        0 <= ny < active.shape[0]
                        and 0 <= nx < active.shape[1]
                        and active[ny, nx]
                        and not seen[ny, nx]
                    ):
                        seen[ny, nx] = True
                        stack.append((ny, nx))
        result.append(component)
    result.sort(
        key=lambda value: (
            -int(value.sum()),
            int(np.argwhere(value)[0, 0]),
            int(np.argwhere(value)[0, 1]),
        )
    )
    return result


def _point(anatomy: Any, descriptor: tuple[str, str]) -> tuple[int, int]:
    kind, name = descriptor
    return tuple(anatomy.joints[name] if kind == "joint" else anatomy.sockets[name])


def _segment_distance_squared(
    points_yx: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
) -> np.ndarray:
    points = np.stack((points_yx[:, 1], points_yx[:, 0]), axis=1).astype(np.float64)
    a = np.asarray(start, dtype=np.float64)
    b = np.asarray(end, dtype=np.float64)
    vector = b - a
    denominator = float(vector @ vector)
    if denominator <= 1e-12:
        delta = points - a
        return np.sum(delta * delta, axis=1)
    t = np.clip(((points - a) @ vector) / denominator, 0.0, 1.0)
    closest = a[None, :] + t[:, None] * vector[None, :]
    delta = points - closest
    return np.sum(delta * delta, axis=1)


def _anchor_specs(anatomy: Any) -> list[tuple[str, str, str, tuple[int, int]]]:
    result = [
        ("joint", name, JOINT_DRIVER[name], tuple(point))
        for name, point in anatomy.joints.items()
    ]
    result.extend(
        ("socket", name, SOCKET_DRIVER[name], tuple(point))
        for name, point in anatomy.sockets.items()
    )
    return result


def derive_logical_projection(
    sample: RepairSourceSample,
) -> tuple[np.ndarray, tuple[RepairAnchor, ...], list[dict[str, Any]], list[dict[str, Any]]]:
    anatomy = derive_conditioned_anatomy(
        sample.corpus_seed,
        family=sample.family_id,
        subtype_id=sample.subtype_id,
        role_id=sample.role_id,
    )
    part = sample.part_owner
    physical = (part != 0) & (part != AURA_OWNER_ID)
    physical_points = np.argwhere(physical)
    if len(physical_points) < len(DRIVER_NAMES) * REPAIR_MIN_DRIVER_PIXELS:
        raise ValueError("Neural repair cannot allocate the minimum logical driver support")

    distances: dict[str, np.ndarray] = {}
    for driver in DRIVER_NAMES:
        first, second = DRIVER_SEGMENTS[driver]
        distances[driver] = _segment_distance_squared(
            physical_points,
            _point(anatomy, first),
            _point(anatomy, second),
        )

    result = np.full(part.shape, BACKGROUND_DRIVER, dtype=np.uint8)
    reserved = np.zeros(part.shape, dtype=bool)

    # Reserve one nearest physical support for every named anchor. This retains
    # exact points whenever possible while resolving cross-driver co-location
    # without changing a neural tuple.
    for _kind, _name, driver, source_point in _anchor_specs(anatomy):
        available = physical_points[~reserved[physical_points[:, 0], physical_points[:, 1]]]
        if not len(available):
            raise ValueError("Neural repair exhausted physical anchor support")
        dx = available[:, 1].astype(np.float64) - float(source_point[0])
        dy = available[:, 0].astype(np.float64) - float(source_point[1])
        distance = dx * dx + dy * dy
        candidates = available[distance == distance.min()]
        order = np.lexsort((candidates[:, 1], candidates[:, 0]))
        y, x = map(int, candidates[order[0]])
        result[y, x] = DRIVER_INDEX[driver]
        reserved[y, x] = True

    inverse_direct: dict[str, set[int]] = {driver: set() for driver in DRIVER_NAMES}
    for owner_id, driver in DIRECT_OWNER_DRIVER.items():
        inverse_direct[driver].add(owner_id)

    for driver in DRIVER_NAMES:
        driver_id = DRIVER_INDEX[driver]
        needed = max(
            0,
            max(MIN_DRIVER_PIXELS[driver], REPAIR_MIN_DRIVER_PIXELS)
            - int((result == driver_id).sum()),
        )
        if not needed:
            continue
        available_mask = physical & ~reserved
        preferred_mask = available_mask & np.isin(part, sorted(inverse_direct[driver]))
        for mask in (preferred_mask, available_mask):
            if needed <= 0:
                break
            candidates = np.argwhere(mask)
            if not len(candidates):
                continue
            score = _segment_distance_squared(
                candidates,
                _point(anatomy, DRIVER_SEGMENTS[driver][0]),
                _point(anatomy, DRIVER_SEGMENTS[driver][1]),
            )
            order = np.lexsort((candidates[:, 1], candidates[:, 0], score))
            for index in order[:needed]:
                y, x = map(int, candidates[index])
                result[y, x] = driver_id
                reserved[y, x] = True
                preferred_mask[y, x] = False
                available_mask[y, x] = False
                needed -= 1
                if needed == 0:
                    break
        if needed:
            raise ValueError(f"Neural repair could not seed driver {driver}")

    for y, x in np.argwhere(physical & ~reserved):
        owner = int(part[y, x])
        direct = DIRECT_OWNER_DRIVER.get(owner)
        if direct is not None:
            result[y, x] = DRIVER_INDEX[direct]
            continue
        point = np.asarray([[y, x]], dtype=np.int64)
        score = [
            float(
                _segment_distance_squared(
                    point,
                    _point(anatomy, DRIVER_SEGMENTS[driver][0]),
                    _point(anatomy, DRIVER_SEGMENTS[driver][1]),
                )[0]
            )
            for driver in DRIVER_NAMES
        ]
        result[y, x] = int(np.argmin(score))

    # Aura is an effect layer, never physical support. Attach every aura tuple
    # to its nearest already-assigned physical driver for coherent motion.
    assigned_physical = np.argwhere(physical)
    for y, x in np.argwhere(part == AURA_OWNER_ID):
        distance = (
            (assigned_physical[:, 0] - y) ** 2
            + (assigned_physical[:, 1] - x) ** 2
        )
        candidates = assigned_physical[distance == distance.min()]
        candidate_drivers = result[candidates[:, 0], candidates[:, 1]]
        order = np.lexsort((candidates[:, 1], candidates[:, 0], candidate_drivers))
        cy, cx = map(int, candidates[order[0]])
        result[y, x] = result[cy, cx]

    anchors: list[RepairAnchor] = []
    for kind, name, driver, source_point in _anchor_specs(anatomy):
        candidates = np.argwhere(physical & (result == DRIVER_INDEX[driver]))
        dx = candidates[:, 1].astype(np.float64) - float(source_point[0])
        dy = candidates[:, 0].astype(np.float64) - float(source_point[1])
        distance = dx * dx + dy * dy
        nearest = candidates[distance == distance.min()]
        order = np.lexsort((nearest[:, 1], nearest[:, 0]))
        y, x = map(int, nearest[order[0]])
        displacement = math.sqrt(float(distance.min()))
        if displacement > MAX_ANCHOR_DISPLACEMENT:
            raise ValueError(
                f"Logical anchor {kind}.{name} requires displacement {displacement:.3f}"
            )
        anchors.append(
            RepairAnchor(
                name=name,
                kind=kind,
                driver=driver,
                source_point=source_point,
                point=(x, y),
                support_point=(x, y),
                displacement=displacement,
                observed_owner=int(part[y, x]),
                policy=(
                    "exact-conditioned-point"
                    if (x, y) == source_point
                    else "nearest-existing-physical-driver-support"
                ),
            )
        )

    components = _components(physical)
    component_records: list[dict[str, Any]] = []
    for index, component in enumerate(components):
        driver_counts = [
            int(((result == driver_id) & component).sum())
            for driver_id in range(len(DRIVER_NAMES))
        ]
        component_records.append(
            {
                "component": index,
                "pixel_count": int(component.sum()),
                "dominant_driver": DRIVER_NAMES[int(np.argmax(driver_counts))],
                "physical": True,
            }
        )

    logical_links: list[dict[str, Any]] = []
    dominant = np.argwhere(components[0])
    for index, component in enumerate(components[1:], start=1):
        detached = np.argwhere(component)
        delta = detached[:, None, :] - dominant[None, :, :]
        squared = np.sum(delta * delta, axis=2)
        source_index, target_index = np.unravel_index(int(np.argmin(squared)), squared.shape)
        source_y, source_x = map(int, detached[source_index])
        target_y, target_x = map(int, dominant[target_index])
        logical_links.append(
            {
                "component": index,
                "source_point": [source_x, source_y],
                "target_component": 0,
                "target_point": [target_x, target_y],
                "distance": round(math.sqrt(float(squared[source_index, target_index])), 9),
                "joint_type": "orbital" if sample.family == "anomaly" else "fixed-logical",
                "pixels_inserted": 0,
            }
        )
    return (
        readonly_array(result, dtype=np.uint8),
        tuple(anchors),
        component_records,
        logical_links,
    )


def _categorize(reason: str) -> str:
    if "lands on neural background" in reason:
        return "anchor_on_background"
    if "physical rig has" in reason and "components" in reason:
        return "plant_topology"
    if "required " in reason and " owner is absent" in reason:
        return "required_owner_absence"
    if "foreground pixels violate the 3-pixel margin" in reason:
        return "safety_margin"
    raise ValueError(f"Unclassified frozen-bridge rejection: {reason}")


def frozen_bridge_baseline(sample: RepairSourceSample) -> dict[str, Any]:
    if binder_source_hash() != EXPECTED_BRIDGE_SOURCE_SHA256:
        raise ValueError("Frozen neural rig bridge source hash drifted")
    try:
        binding = bind_raw_sample_archive(
            sample.raw_archive_path,
            raw_manifest_path=sample.raw_manifest_path,
            legal_tuples=sample.legal_tuples,
        )
    except BindingRejected as error:
        reasons = list(error.errors)
        categories = []
        for reason in reasons:
            category = _categorize(reason)
            if category not in categories:
                categories.append(category)
        return {
            "status": "rejected",
            "binding_sha256": None,
            "categories": categories,
            "reasons": reasons,
        }
    return {
        "status": "accepted",
        "binding_sha256": binding.sha256,
        "categories": [],
        "reasons": [],
    }


def compile_repair_plan(source: RepairSource, sample: RepairSourceSample) -> dict[str, Any]:
    baseline = frozen_bridge_baseline(sample)
    driver_index, anchors, components, logical_links = derive_logical_projection(sample)
    missing_owners = [
        name for name, owner_id in REQUIRED_OWNER_IDS.items() if not bool((sample.part_owner == owner_id).any())
    ]
    unsafe = np.zeros(sample.part_owner.shape, dtype=bool)
    unsafe[:3] = True
    unsafe[-3:] = True
    unsafe[:, :3] = True
    unsafe[:, -3:] = True
    unsafe_foreground = int(((sample.part_owner != 0) & unsafe).sum())
    unsafe_physical = int(
        ((sample.part_owner != 0) & (sample.part_owner != AURA_OWNER_ID) & unsafe).sum()
    )
    unsafe_aura = int(((sample.part_owner == AURA_OWNER_ID) & unsafe).sum())
    operations = [
        "logical_driver_projection_v2",
        "nonphysical_aura_attachment_v1",
        "per_clip_motion_envelope_fit_v1",
    ]
    for category in baseline["categories"]:
        operations.append(
            {
                "anchor_on_background": "anchor_support_reselection_v1",
                "plant_topology": "logical_component_link_v1",
                "required_owner_absence": "guide_anchor_owner_fallback_v1",
                "safety_margin": "aura_margin_deferred_to_clip_fit_v1",
            }[category]
        )
    anchor_records = [anchor.metadata() for anchor in anchors]
    base = {
        "format": PLAN_FORMAT,
        "status": "ready",
        "repair_version": REPAIR_VERSION,
        "sample_id": sample.sample_id,
        "ordinal": sample.ordinal,
        "condition": {
            "family": sample.family,
            "family_id": sample.family_id,
            "subtype_id": sample.subtype_id,
            "role_id": sample.role_id,
            "corpus_seed": sample.corpus_seed,
            "sample_seed": sample.sample_seed,
        },
        "source": {
            "generation_manifest": {
                "path": _relative(source.generation_manifest_path),
                "bytes": source.generation_manifest_bytes,
                "sha256": source.generation_manifest_sha256,
            },
            "style_manifest": {
                "path": _relative(source.style_manifest_path),
                "bytes": source.style_manifest_bytes,
                "sha256": source.style_manifest_sha256,
            },
            "raw_manifest": {
                "path": _relative(sample.raw_manifest_path),
                "bytes": sample.raw_manifest_bytes,
                "sha256": sample.raw_manifest_sha256,
            },
            "raw_archive": {
                "path": _relative(sample.raw_archive_path),
                "bytes": sample.raw_archive_bytes,
                "sha256": sample.raw_archive_sha256,
            },
            "raw_fields_sha256": sample.raw_fields_sha256,
            "compiled_fields_sha256": sample.compiled_fields_sha256,
            "static_palette_sha256": sample.static_palette_sha256,
            "legal_tuple_fingerprint": source.legal_tuple_fingerprint,
            "frozen_bridge_source_sha256": EXPECTED_BRIDGE_SOURCE_SHA256,
        },
        "baseline_v1": baseline,
        "repair": {
            "operations": operations,
            "driver_policy": "robust-minimum-12px-segment-voronoi-v2",
            "anchor_policy": "nearest-existing-physical-driver-support-v1",
            "aura_policy": "nonphysical-nearest-physical-driver-effect-attachment-v1",
            "component_policy": "preserve-pixels-add-logical-links-only-v1",
            "motion_envelope_policy": "clip-wide-facing-preserving-bounded-attenuation-v1",
            "required_owner_fallbacks": missing_owners,
            "anchors": anchor_records,
            "physical_components": components,
            "logical_links": logical_links,
            "rest_margin": {
                "unsafe_foreground_pixels": unsafe_foreground,
                "unsafe_physical_pixels": unsafe_physical,
                "unsafe_aura_pixels": unsafe_aura,
                "physical_margin_clear": unsafe_physical == 0,
                "aura_deferred_to_motion_envelope": unsafe_aura > 0,
            },
        },
        "expected": {
            "driver_index_sha256": array_sha256("repair_driver_index", driver_index),
            "physical_component_count": len(components),
            "logical_link_count": len(logical_links),
            "maximum_anchor_displacement": round(
                max(anchor.displacement for anchor in anchors), 9
            ),
            "minimum_driver_pixels": REPAIR_MIN_DRIVER_PIXELS,
        },
        "gates": {
            "raw_rest_arrays_unchanged": True,
            "raw_and_compiled_fields_identical": True,
            "no_pixels_inserted_removed_translated_cropped_or_relabeled_at_rest": True,
            "anchors_use_existing_physical_pixels": True,
            "all_drivers_have_minimum_support": True,
            "aura_is_nonphysical_effect_metadata": True,
            "disconnected_components_use_logical_links_only": True,
            "motion_fit_is_clip_local_and_rest_preserving": True,
            "source_artifacts_hash_bound": True,
        },
    }
    base["hashes"] = {"plan_sha256": sha256_bytes(canonical_json_bytes(base))}
    return base


def load_repair_plan(path: Path) -> dict[str, Any]:
    plan = load_schema_json(
        path,
        maximum_bytes=MAX_PLAN_BYTES,
        label="neural rig repair plan",
        schema=PLAN_SCHEMA,
    )
    unsigned = dict(plan)
    hashes = unsigned.pop("hashes")
    if hashes["plan_sha256"] != sha256_bytes(canonical_json_bytes(unsigned)):
        raise ValueError("Neural rig repair plan self-hash mismatch")
    return plan
