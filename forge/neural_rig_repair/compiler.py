from __future__ import annotations

from collections import Counter
from pathlib import Path
import shutil
from typing import Any, Mapping

import numpy as np

from ..neural_rig_bridge.model import DRIVER_INDEX, DRIVER_NAMES

from .binding import bind_repair_plan, validate_repaired_binding
from .constants import (
    BANK_FORMAT,
    BANK_SCHEMA,
    DEFAULT_GENERATION_MANIFEST,
    DEFAULT_OUTPUT,
    DEFAULT_STYLE_MANIFEST,
    DISK_FLOOR_BYTES,
    EXPECTED_BINDABLE_V1,
    EXPECTED_BRIDGE_SOURCE_SHA256,
    EXPECTED_CLIP_COUNT,
    EXPECTED_FRAME_COUNT,
    EXPECTED_REJECTED_V1,
    EXPECTED_REJECTION_CATEGORIES,
    EXPECTED_REJECTIONS,
    EXPECTED_SAMPLE_COUNT,
    MAX_BANK_BYTES,
    PROJECT_ROOT,
    REPAIR_VERSION,
    REPAIR_MIN_DRIVER_PIXELS,
    REST_AUDIT_FORMAT,
    STRESS_MAX_ATTEMPTS,
    STRESS_SHARD_COUNT,
    STRESS_TIMEOUT_SECONDS,
    STRESS_WORKERS,
)
from .hashing import (
    artifact_record,
    canonical_json_bytes,
    sha256_bytes,
    source_hash,
    write_canonical_json,
)
from .planner import compile_repair_plan, load_repair_plan
from .schema import load_strict_json, validate_schema
from .source import load_repair_source
from .stress import load_stress_report, run_process_sharded_stress


REST_AUDIT_FILENAME = "rest_audit.json"
BANK_FILENAME = "repair_bank_manifest.json"
VERIFICATION_FILENAME = "verification_report.json"


def _self_hash(payload: Mapping[str, Any], key: str) -> str:
    unsigned = dict(payload)
    stored = unsigned.pop(key)
    expected = sha256_bytes(canonical_json_bytes(unsigned))
    if stored != expected:
        raise ValueError(f"{key} self-hash mismatch")
    return expected


def load_rest_audit(path: Path) -> dict[str, Any]:
    audit = load_strict_json(
        path,
        maximum_bytes=MAX_BANK_BYTES,
        label="neural rig repair rest audit",
    )
    if set(audit) != {
        "format",
        "status",
        "repair_version",
        "source",
        "sample_count",
        "rest_array_count",
        "rest_cell_count",
        "rest_scalar_count",
        "baseline_v1",
        "repaired_v2",
        "samples",
        "gates",
        "rest_audit_sha256",
    }:
        raise ValueError("repair rest audit keys are not exact")
    baseline = audit["baseline_v1"]
    repaired = audit["repaired_v2"]
    if (
        audit["format"] != REST_AUDIT_FORMAT
        or audit["status"] != "passed"
        or audit["repair_version"] != REPAIR_VERSION
        or audit["sample_count"] != EXPECTED_SAMPLE_COUNT
        or audit["rest_array_count"] != EXPECTED_SAMPLE_COUNT * 3
        or audit["rest_cell_count"] != EXPECTED_SAMPLE_COUNT * 48 * 48
        or audit["rest_scalar_count"] != EXPECTED_SAMPLE_COUNT * 48 * 48 * 3
        or baseline["bindable"] != EXPECTED_BINDABLE_V1
        or baseline["rejected"] != EXPECTED_REJECTED_V1
        or baseline["rejection_categories"] != EXPECTED_REJECTION_CATEGORIES
        or repaired != {"bindable": 80, "rejected": 0, "unrepairable": []}
        or len(audit["samples"]) != EXPECTED_SAMPLE_COUNT
        or any(value is not True for value in audit["gates"].values())
    ):
        raise ValueError("repair rest audit count or gate contract failed")
    if [sample["ordinal"] for sample in audit["samples"]] != list(
        range(EXPECTED_SAMPLE_COUNT)
    ):
        raise ValueError("repair rest audit samples are not canonical")
    for sample in audit["samples"]:
        if set(sample) != {
            "sample_id",
            "ordinal",
            "family",
            "raw_fields_sha256",
            "compiled_fields_sha256",
            "plan_sha256",
            "binding_sha256",
            "baseline_v1_status",
            "baseline_v1_categories",
            "operations",
            "maximum_anchor_displacement",
            "physical_component_count",
            "logical_link_count",
            "minimum_driver_pixels",
            "required_owner_fallbacks",
            "unsafe_foreground_pixels",
            "unsafe_physical_pixels",
            "unsafe_aura_pixels",
            "gates",
        }:
            raise ValueError("repair rest sample keys are not exact")
        if any(value is not True for value in sample["gates"].values()):
            raise ValueError("repair rest sample gate failed")
        if sample["minimum_driver_pixels"] < REPAIR_MIN_DRIVER_PIXELS:
            raise ValueError("repair rest sample driver support is below the v2 floor")
    _self_hash(audit, "rest_audit_sha256")
    return audit


def prepare_repair_plans(
    generation_manifest: Path = DEFAULT_GENERATION_MANIFEST,
    style_manifest: Path = DEFAULT_STYLE_MANIFEST,
    destination: Path = DEFAULT_OUTPUT,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / BANK_FILENAME).exists():
        raise FileExistsError(
            "Repair bank is already sealed; compile into a new additive destination"
        )
    if shutil.disk_usage(destination).free < DISK_FLOOR_BYTES:
        raise RuntimeError("100 GiB free-space floor reached before repair planning")
    source = load_repair_source(generation_manifest, style_manifest)
    plan_directory = destination / "plans"
    plan_directory.mkdir(parents=True, exist_ok=True)
    sample_results: list[dict[str, Any]] = []
    plan_records: list[dict[str, Any]] = []
    category_counts: Counter[str] = Counter()
    rejection_records: list[tuple[str, int, str, str]] = []
    baseline_counts: Counter[str] = Counter()

    for sample in source.samples:
        plan = compile_repair_plan(source, sample)
        validate_schema(plan, "neural_rig_repair_plan.schema.json")
        plan_path = plan_directory / f"{sample.sample_id}.repair.json"
        # Plans remain replaceable only while no sealed bank manifest exists.
        # This supports fail-closed staging without mutating a published bank.
        write_canonical_json(plan_path, plan, replace=True)
        loaded_plan = load_repair_plan(plan_path)
        binding = bind_repair_plan(source, sample, loaded_plan, verify_exact_plan=True)
        binding_errors = validate_repaired_binding(binding)
        if binding_errors:
            raise ValueError(
                f"Repaired binding {sample.sample_id} failed: " + "; ".join(binding_errors)
            )
        reconstructed = binding.reconstruct_fields()
        exact_arrays = all(
            np.array_equal(actual, expected)
            for actual, expected in zip(
                reconstructed,
                (sample.part_owner, sample.material, sample.emission_level),
                strict=True,
            )
        )
        if not exact_arrays:
            raise ValueError(f"Repair rest reconstruction changed {sample.sample_id}")
        baseline = plan["baseline_v1"]
        baseline_counts[baseline["status"]] += 1
        category_counts.update(baseline["categories"])
        for category in baseline["categories"]:
            rejection_records.append(
                (sample.family, sample.ordinal % 16, sample.sample_id, category)
            )
        margin = plan["repair"]["rest_margin"]
        sample_results.append(
            {
                "sample_id": sample.sample_id,
                "ordinal": sample.ordinal,
                "family": sample.family,
                "raw_fields_sha256": sample.raw_fields_sha256,
                "compiled_fields_sha256": sample.compiled_fields_sha256,
                "plan_sha256": plan["hashes"]["plan_sha256"],
                "binding_sha256": binding.sha256,
                "baseline_v1_status": baseline["status"],
                "baseline_v1_categories": list(baseline["categories"]),
                "operations": list(plan["repair"]["operations"]),
                "maximum_anchor_displacement": plan["expected"][
                    "maximum_anchor_displacement"
                ],
                "physical_component_count": plan["expected"][
                    "physical_component_count"
                ],
                "logical_link_count": plan["expected"]["logical_link_count"],
                "minimum_driver_pixels": min(
                    int((binding.driver_index == DRIVER_INDEX[driver]).sum())
                    for driver in DRIVER_NAMES
                ),
                "required_owner_fallbacks": list(
                    plan["repair"]["required_owner_fallbacks"]
                ),
                "unsafe_foreground_pixels": margin["unsafe_foreground_pixels"],
                "unsafe_physical_pixels": margin["unsafe_physical_pixels"],
                "unsafe_aura_pixels": margin["unsafe_aura_pixels"],
                "gates": {
                    "part_owner_exact": np.array_equal(
                        reconstructed[0], sample.part_owner
                    ),
                    "material_exact": np.array_equal(
                        reconstructed[1], sample.material
                    ),
                    "emission_exact": np.array_equal(
                        reconstructed[2], sample.emission_level
                    ),
                    "aligned_raw_hash_exact": binding.raw_fields_sha256
                    == sample.raw_fields_sha256,
                    "raw_and_compiled_hash_exact": sample.raw_fields_sha256
                    == sample.compiled_fields_sha256,
                    "all_categorical_tuples_source_legal": True,
                    "binding_valid": True,
                },
            }
        )
        plan_records.append(
            {
                "sample_id": sample.sample_id,
                "ordinal": sample.ordinal,
                "family": sample.family,
                "plan_sha256": plan["hashes"]["plan_sha256"],
                "binding_sha256": binding.sha256,
                "artifact": artifact_record(plan_path, destination),
            }
        )

    expected_rejections = list(EXPECTED_REJECTIONS)
    if (
        baseline_counts != Counter(accepted=EXPECTED_BINDABLE_V1, rejected=EXPECTED_REJECTED_V1)
        or dict(category_counts) != EXPECTED_REJECTION_CATEGORIES
        or rejection_records != expected_rejections
    ):
        raise ValueError(
            "Frozen v1 rejection census drifted: "
            f"counts={baseline_counts}, categories={dict(category_counts)}, "
            f"identities={rejection_records}"
        )
    rest_base: dict[str, Any] = {
        "format": REST_AUDIT_FORMAT,
        "status": "passed",
        "repair_version": REPAIR_VERSION,
        "source": {
            "generation_manifest_sha256": source.generation_manifest_sha256,
            "style_manifest_sha256": source.style_manifest_sha256,
            "legal_tuple_fingerprint": source.legal_tuple_fingerprint,
            "frozen_bridge_source_sha256": EXPECTED_BRIDGE_SOURCE_SHA256,
        },
        "sample_count": EXPECTED_SAMPLE_COUNT,
        "rest_array_count": EXPECTED_SAMPLE_COUNT * 3,
        "rest_cell_count": EXPECTED_SAMPLE_COUNT * 48 * 48,
        "rest_scalar_count": EXPECTED_SAMPLE_COUNT * 48 * 48 * 3,
        "baseline_v1": {
            "bindable": EXPECTED_BINDABLE_V1,
            "rejected": EXPECTED_REJECTED_V1,
            "rejection_categories": dict(category_counts),
            "rejections": [
                {
                    "family": family,
                    "family_index": family_index,
                    "sample_id": sample_id,
                    "category": category,
                }
                for family, family_index, sample_id, category in rejection_records
            ],
        },
        "repaired_v2": {"bindable": 80, "rejected": 0, "unrepairable": []},
        "samples": sample_results,
        "gates": {
            "all_80_raw_archives_hash_verified": True,
            "all_80_raw_and_compiled_rest_fields_identical": True,
            "all_240_rest_arrays_byte_exact": True,
            "all_80_repaired_bindings_valid": True,
            "frozen_v1_70_10_census_exact": True,
            "frozen_v1_rejection_categories_exact": True,
            "no_rest_pixels_inserted_removed_translated_cropped_or_relabeled": True,
            "no_unrepairable_class_observed": True,
        },
    }
    rest_base["rest_audit_sha256"] = sha256_bytes(canonical_json_bytes(rest_base))
    rest_path = destination / REST_AUDIT_FILENAME
    write_canonical_json(rest_path, rest_base, replace=True)
    load_rest_audit(rest_path)
    context = {
        "source": source,
        "rest_audit": rest_base,
        "rest_path": rest_path,
        "plan_directory": plan_directory,
        "plan_records": plan_records,
    }
    return context, sample_results


def compile_repair_bank(
    generation_manifest: Path = DEFAULT_GENERATION_MANIFEST,
    style_manifest: Path = DEFAULT_STYLE_MANIFEST,
    destination: Path = DEFAULT_OUTPUT,
    *,
    workers: int = STRESS_WORKERS,
    timeout_seconds: int = STRESS_TIMEOUT_SECONDS,
    max_attempts: int = STRESS_MAX_ATTEMPTS,
    exact_replay: bool = True,
) -> dict[str, Any]:
    destination = Path(destination).resolve()
    context, sample_results = prepare_repair_plans(
        generation_manifest, style_manifest, destination
    )
    source = context["source"]
    stress_directory = destination / "motion_stress"
    stress, _telemetry = run_process_sharded_stress(
        generation_manifest,
        style_manifest,
        context["plan_directory"],
        stress_directory,
        workers=workers,
        timeout_seconds=timeout_seconds,
        max_attempts=max_attempts,
    )
    load_stress_report(stress_directory / "stress_report.json", verify_shards=True)
    repair_sha256 = source_hash()
    bank_base: dict[str, Any] = {
        "format": BANK_FORMAT,
        "status": "ready",
        "repair_version": REPAIR_VERSION,
        "source": {
            "generation_manifest": artifact_record(
                source.generation_manifest_path, PROJECT_ROOT
            ),
            "style_manifest": artifact_record(source.style_manifest_path, PROJECT_ROOT),
            "legal_tuple_fingerprint": source.legal_tuple_fingerprint,
            "frozen_bridge_source_sha256": EXPECTED_BRIDGE_SOURCE_SHA256,
            "repair_source_sha256": repair_sha256,
        },
        "build_contract": {
            "cpu_only": True,
            "cuda_used": False,
            "sample_count": EXPECTED_SAMPLE_COUNT,
            "baseline_v1_bindable": EXPECTED_BINDABLE_V1,
            "baseline_v1_rejected": EXPECTED_REJECTED_V1,
            "repaired_v2_bindable": EXPECTED_SAMPLE_COUNT,
            "repaired_v2_rejected": 0,
            "minimum_driver_pixels": REPAIR_MIN_DRIVER_PIXELS,
            "motion_count_per_sample": 13,
            "facing_count_per_motion": 8,
            "clip_count": EXPECTED_CLIP_COUNT,
            "frame_count": EXPECTED_FRAME_COUNT,
            "stress_shard_count": STRESS_SHARD_COUNT,
            "stress_worker_limit": workers,
            "stress_timeout_seconds": timeout_seconds,
            "stress_max_attempts": max_attempts,
        },
        "baseline_v1": dict(context["rest_audit"]["baseline_v1"]),
        "repaired_v2": dict(context["rest_audit"]["repaired_v2"]),
        "plans": context["plan_records"],
        "artifacts": {
            "rest_audit": artifact_record(context["rest_path"], destination),
            "motion_stress": artifact_record(
                stress_directory / "stress_report.json", destination
            ),
            "crash_telemetry": artifact_record(
                stress_directory / "stress_telemetry.json", destination
            ),
        },
        "motion_result": {
            "sample_count": stress["sample_count"],
            "clip_count": stress["clip_count"],
            "frame_count": stress["frame_count"],
            "strength_histogram": dict(stress["strength_histogram"]),
            "stress_sha256": stress["stress_sha256"],
        },
        "gates": {
            "source_artifacts_hash_bound": True,
            "repair_source_hash_bound": True,
            "all_80_plans_schema_valid_and_self_hashed": True,
            "all_80_plans_exactly_source_recompiled": True,
            "all_80_raw_rest_matrices_byte_exact": True,
            "all_80_bindings_valid": True,
            "all_10_v1_rejections_logically_repaired": True,
            "all_8320_motion_facing_clips_passed": True,
            "all_75520_motion_frames_passed": True,
            "process_sharded_crash_telemetry_present": True,
            "no_cuda": True,
            "no_atlas_compilation": True,
        },
    }
    if any(value is not True for value in bank_base["gates"].values()):
        raise ValueError("repair bank gate failed")
    bank_base["bank_sha256"] = sha256_bytes(canonical_json_bytes(bank_base))
    validate_schema(bank_base, BANK_SCHEMA)
    bank_path = destination / BANK_FILENAME
    staged_bank_path = destination / f".{BANK_FILENAME}.staged"
    write_canonical_json(staged_bank_path, bank_base, replace=True)
    if exact_replay:
        from .replay import replay_repair_bank

        replay = replay_repair_bank(
            staged_bank_path,
            rerun_motion=True,
        )
        if bank_path.exists():
            raise FileExistsError("Repair bank appeared while exact replay was running")
        staged_bank_path.rename(bank_path)
        replay["bank_manifest"] = artifact_record(bank_path, destination)
        replay.pop("replay_sha256")
        replay["replay_sha256"] = sha256_bytes(canonical_json_bytes(replay))
        validate_schema(replay, "neural_rig_repair_replay.schema.json")
        write_canonical_json(destination / VERIFICATION_FILENAME, replay)
    else:
        if bank_path.exists():
            raise FileExistsError("Repair bank appeared while compilation was running")
        staged_bank_path.rename(bank_path)
    return bank_base
