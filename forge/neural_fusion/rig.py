from __future__ import annotations

from types import MappingProxyType

import numpy as np

from ..morphology.constants import PART_OWNER_NAMES
from ..neural_rig_repair.binding import validate_repaired_binding
from ..neural_rig_repair.hashing import array_sha256, canonical_json_bytes, sha256_bytes
from ..neural_rig_repair.model import RepairSourceSample, RepairedRigBinding, readonly_array
from ..neural_rig_repair.planner import derive_logical_projection
from .model import FusionSpecimen


def build_fusion_binding(specimen: FusionSpecimen) -> RepairedRigBinding:
    condition = specimen.genome.condition
    failures: list[str] = []
    projection = None
    rig_seed = specimen.genome.seed
    rig_seed_attempt = 0
    # A decoded or aggressively fused silhouette need not align with the
    # procedural anatomy implied by its first identity seed. Search a small,
    # finite, lineage-derived seed family for an anatomy whose named anchors
    # all land within the existing 12-pixel support domain. Fields never
    # change; this chooses only logical pivots and is recorded in the binding.
    for attempt in range(64):
        candidate_seed = int(
            (specimen.genome.seed + attempt * 0x9E3779B1) & 0x7FFFFFFFFFFFFFFF
        )
        sample = RepairSourceSample(
            sample_id=specimen.genome.specimen_id,
            ordinal=condition.ordinal,
            family=condition.morphology_name,
            family_id=condition.morphology_id,
            subtype_id=condition.subtype_id,
            role_id=condition.role_id,
            corpus_seed=candidate_seed,
            sample_seed=condition.sample_seed,
            part_owner=specimen.part_owner,
            material=specimen.material,
            emission_level=specimen.emission_level,
            guide=specimen.guide,
            genes=specimen.genes,
            legal_tuples=specimen.legal_tuples,
            raw_manifest_path=__file__,
            raw_manifest_bytes=0,
            raw_manifest_sha256="0" * 64,
            raw_archive_path=__file__,
            raw_archive_bytes=0,
            raw_archive_sha256="0" * 64,
            raw_fields_sha256=specimen.fields_sha256,
            compiled_fields_sha256=specimen.fields_sha256,
            static_palette_sha256="0" * 64,
        )
        try:
            projection = derive_logical_projection(sample)
            rig_seed = candidate_seed
            rig_seed_attempt = attempt
            break
        except ValueError as error:
            failures.append(str(error)[:300])
    if projection is None:
        raise ValueError(
            "neural fusion exhausted 64 bounded logical anatomy seeds: "
            + " | ".join(failures[-4:])
        )
    driver_index, anchors, components, logical_links = projection
    joints = {anchor.name: anchor for anchor in anchors if anchor.kind == "joint"}
    sockets = {anchor.name: anchor for anchor in anchors if anchor.kind == "socket"}
    owner_masks = np.stack(
        [specimen.part_owner == owner for owner in range(1, len(PART_OWNER_NAMES))]
    ).astype(np.uint8)
    plan_base = {
        "format": "nullvector-neural-fusion-rig-plan-v1",
        "specimen_id": specimen.genome.specimen_id,
        "lineage_sha256": specimen.genome.lineage_sha256,
        "fields_sha256": specimen.fields_sha256,
        "rig_seed": rig_seed,
        "rig_seed_attempt": rig_seed_attempt,
        "rig_seed_policy": "bounded-lineage-derived-logical-anatomy-search-v1",
        "expected": {
            "driver_index_sha256": array_sha256("repair_driver_index", driver_index),
            "physical_component_count": len(components),
            "logical_link_count": len(logical_links),
        },
        "gates": {
            "fusion_fields_immutable": True,
            "logical_projection_only": True,
            "no_pixel_substitution": True,
            "all_drivers_supported": True,
        },
    }
    plan_base["hashes"] = {"plan_sha256": sha256_bytes(canonical_json_bytes(plan_base))}
    manifest = {
        "format": "nullvector-neural-fusion-rig-binding-v1",
        "sample_id": specimen.genome.specimen_id,
        "condition": condition.as_dict(),
        "lineage_sha256": specimen.genome.lineage_sha256,
        "rig_seed": rig_seed,
        "rig_seed_attempt": rig_seed_attempt,
        "plan_sha256": plan_base["hashes"]["plan_sha256"],
        "raw_fields_sha256": specimen.fields_sha256,
        "driver_index_sha256": array_sha256("repair_driver_index", driver_index),
        "owner_masks_sha256": array_sha256("repair_owner_masks", owner_masks),
        "joints": {name: anchor.metadata() for name, anchor in joints.items()},
        "sockets": {name: anchor.metadata() for name, anchor in sockets.items()},
        "physical_components": components,
        "logical_links": logical_links,
        "gates": dict(plan_base["gates"]),
    }
    manifest["binding_sha256"] = sha256_bytes(canonical_json_bytes(manifest))
    binding = RepairedRigBinding(
        sample_id=specimen.genome.specimen_id,
        family=condition.morphology_name,
        family_id=condition.morphology_id,
        subtype_id=condition.subtype_id,
        role_id=condition.role_id,
        part_owner=readonly_array(specimen.part_owner, dtype=np.uint8),
        material=readonly_array(specimen.material, dtype=np.uint8),
        emission_level=readonly_array(specimen.emission_level, dtype=np.uint8),
        guide=readonly_array(specimen.guide, dtype=np.float32),
        genes=readonly_array(specimen.genes, dtype=np.float32),
        legal_tuples=readonly_array(specimen.legal_tuples, dtype=np.uint8),
        driver_index=readonly_array(driver_index, dtype=np.uint8),
        owner_masks=readonly_array(owner_masks, dtype=np.uint8),
        joints=MappingProxyType(joints),
        sockets=MappingProxyType(sockets),
        plan=MappingProxyType(plan_base),
        manifest=MappingProxyType(manifest),
    )
    errors = validate_repaired_binding(binding)
    if errors:
        raise ValueError("invalid neural fusion binding: " + "; ".join(errors))
    return binding
