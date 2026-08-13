from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from ..morphology.constants import PART_OWNER_NAMES
from ..neural_rig_bridge.hashing import aligned_fields_hash
from ..neural_rig_bridge.model import (
    BACKGROUND_DRIVER,
    DRIVER_INDEX,
    DRIVER_NAMES,
    JOINT_DRIVER,
    MIN_DRIVER_PIXELS,
    SOCKET_DRIVER,
)
from .constants import AURA_OWNER_ID, REPAIR_MIN_DRIVER_PIXELS, REPAIR_VERSION
from .hashing import array_sha256, canonical_json_bytes, sha256_bytes
from .model import RepairAnchor, RepairedRigBinding, RepairSource, RepairSourceSample, readonly_array
from .planner import compile_repair_plan, derive_logical_projection


def bind_repair_plan(
    source: RepairSource,
    sample: RepairSourceSample,
    plan: Mapping[str, Any],
    *,
    verify_exact_plan: bool = True,
) -> RepairedRigBinding:
    plan_payload = dict(plan)
    if verify_exact_plan:
        expected = compile_repair_plan(source, sample)
        if canonical_json_bytes(plan_payload) != canonical_json_bytes(expected):
            raise ValueError(f"Repair plan is not the exact source-derived plan for {sample.sample_id}")
    if plan_payload.get("sample_id") != sample.sample_id:
        raise ValueError("Repair plan sample identity mismatch")
    driver_index, anchors, components, logical_links = derive_logical_projection(sample)
    if plan_payload["expected"]["driver_index_sha256"] != array_sha256(
        "repair_driver_index", driver_index
    ):
        raise ValueError("Repair plan driver projection hash mismatch")
    anchor_map = {(anchor.kind, anchor.name): anchor for anchor in anchors}
    joints = {
        name: anchor_map[("joint", name)] for name in JOINT_DRIVER
    }
    sockets = {
        name: anchor_map[("socket", name)] for name in SOCKET_DRIVER
    }
    owner_masks = np.stack(
        [sample.part_owner == owner for owner in range(1, len(PART_OWNER_NAMES))]
    ).astype(np.uint8)
    manifest_base = {
        "format": "nullvector-neural-rig-repaired-binding-v1",
        "repair_version": REPAIR_VERSION,
        "sample_id": sample.sample_id,
        "condition": dict(plan_payload["condition"]),
        "plan_sha256": plan_payload["hashes"]["plan_sha256"],
        "raw_fields_sha256": sample.raw_fields_sha256,
        "driver_index_sha256": array_sha256("repair_driver_index", driver_index),
        "owner_masks_sha256": array_sha256("repair_owner_masks", owner_masks),
        "joints": {name: anchor.metadata() for name, anchor in joints.items()},
        "sockets": {name: anchor.metadata() for name, anchor in sockets.items()},
        "physical_components": components,
        "logical_links": logical_links,
        "aura": {
            "owner_id": AURA_OWNER_ID,
            "pixel_count": int((sample.part_owner == AURA_OWNER_ID).sum()),
            "physical": False,
            "policy": "nonphysical-nearest-physical-driver-effect-attachment-v1",
        },
        "gates": dict(plan_payload["gates"]),
    }
    manifest_base["binding_sha256"] = sha256_bytes(canonical_json_bytes(manifest_base))
    binding = RepairedRigBinding(
        sample_id=sample.sample_id,
        family=sample.family,
        family_id=sample.family_id,
        subtype_id=sample.subtype_id,
        role_id=sample.role_id,
        part_owner=readonly_array(sample.part_owner, dtype=np.uint8),
        material=readonly_array(sample.material, dtype=np.uint8),
        emission_level=readonly_array(sample.emission_level, dtype=np.uint8),
        guide=readonly_array(sample.guide, dtype=np.float32),
        genes=readonly_array(sample.genes, dtype=np.float32),
        legal_tuples=readonly_array(sample.legal_tuples, dtype=np.uint8),
        driver_index=readonly_array(driver_index, dtype=np.uint8),
        owner_masks=readonly_array(owner_masks, dtype=np.uint8),
        joints=MappingProxyType(joints),
        sockets=MappingProxyType(sockets),
        plan=plan_payload,
        manifest=manifest_base,
    )
    errors = validate_repaired_binding(binding)
    if errors:
        raise ValueError("Invalid repaired binding: " + "; ".join(errors))
    return binding


def validate_repaired_binding(binding: RepairedRigBinding) -> list[str]:
    errors: list[str] = []
    arrays = (
        ("part_owner", binding.part_owner, np.uint8, (48, 48)),
        ("material", binding.material, np.uint8, (48, 48)),
        ("emission_level", binding.emission_level, np.uint8, (48, 48)),
        ("driver_index", binding.driver_index, np.uint8, (48, 48)),
        ("owner_masks", binding.owner_masks, np.uint8, (16, 48, 48)),
    )
    for name, values, dtype, shape in arrays:
        if not isinstance(values, np.ndarray) or values.dtype != dtype or values.shape != shape:
            errors.append(f"{name} must be {dtype} {shape}")
        elif values.flags.writeable:
            errors.append(f"{name} must be immutable")
    if errors:
        return errors
    foreground = binding.part_owner != 0
    physical = foreground & (binding.part_owner != AURA_OWNER_ID)
    if not np.all(binding.driver_index[~foreground] == BACKGROUND_DRIVER):
        errors.append("background has logical drivers")
    if not np.all(binding.driver_index[foreground] < len(DRIVER_NAMES)):
        errors.append("foreground has invalid logical drivers")
    for driver_id, driver in enumerate(DRIVER_NAMES):
        count = int((binding.driver_index == driver_id).sum())
        if count < max(MIN_DRIVER_PIXELS[driver], REPAIR_MIN_DRIVER_PIXELS):
            errors.append(f"driver {driver} has only {count} pixels")
    for kind, values, expected in (
        ("joint", binding.joints, JOINT_DRIVER),
        ("socket", binding.sockets, SOCKET_DRIVER),
    ):
        if set(values) != set(expected):
            errors.append(f"{kind} registry is not exact")
            continue
        for name, anchor in values.items():
            x, y = anchor.point
            sx, sy = anchor.support_point
            if anchor.driver != expected[name]:
                errors.append(f"{kind}.{name} driver mismatch")
            if not physical[y, x] or not physical[sy, sx]:
                errors.append(f"{kind}.{name} lacks existing physical support")
            if int(binding.driver_index[sy, sx]) != DRIVER_INDEX[anchor.driver]:
                errors.append(f"{kind}.{name} support driver mismatch")
    reconstructed = binding.reconstruct_fields()
    for name, actual, expected in zip(
        ("part", "material", "emission"),
        reconstructed,
        (binding.part_owner, binding.material, binding.emission_level),
        strict=True,
    ):
        if not np.array_equal(actual, expected):
            errors.append(f"rest {name} reconstruction is not exact")
    raw_hash = aligned_fields_hash(
        binding.part_owner, binding.material, binding.emission_level
    )
    if raw_hash != binding.raw_fields_sha256:
        errors.append("rest categorical hash mismatch")
    source_tuples = {
        tuple(map(int, row))
        for row in np.stack(
            (binding.part_owner, binding.material, binding.emission_level), axis=-1
        ).reshape(-1, 3)
    }
    legal = {tuple(map(int, row)) for row in binding.legal_tuples}
    if not source_tuples <= legal:
        errors.append("rest categorical tuples leave the source legal table")
    if binding.plan["expected"]["driver_index_sha256"] != array_sha256(
        "repair_driver_index", binding.driver_index
    ):
        errors.append("repair plan driver hash mismatch")
    manifest = dict(binding.manifest)
    stored_hash = manifest.pop("binding_sha256", None)
    if stored_hash != sha256_bytes(canonical_json_bytes(manifest)):
        errors.append("repaired binding manifest hash mismatch")
    return errors
