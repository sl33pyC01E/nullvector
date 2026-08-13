from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .binding import bind_repair_plan, validate_repaired_binding
from .compiler import load_rest_audit
from .constants import (
    BANK_SCHEMA,
    EXPECTED_BRIDGE_SOURCE_SHA256,
    EXPECTED_CLIP_COUNT,
    EXPECTED_FRAME_COUNT,
    EXPECTED_REJECTIONS,
    EXPECTED_SAMPLE_COUNT,
    MAX_BANK_BYTES,
    MAX_JSON_BYTES,
    MAX_PLAN_BYTES,
    MAX_REPLAY_BYTES,
    MAX_STRESS_REPORT_BYTES,
    MAX_STRESS_SHARD_BYTES,
    PROJECT_ROOT,
    REPLAY_FORMAT,
    REPLAY_SCHEMA,
)
from .hashing import (
    artifact_record,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    source_hash,
    write_canonical_json,
)
from .motion import compile_sample_motion_audit
from .planner import compile_repair_plan, load_repair_plan
from .schema import (
    load_schema_json,
    load_strict_json,
    resolve_artifact_record,
    validate_schema,
)
from .source import load_repair_source
from .stress import load_stress_report, load_stress_shard


def _verify_self_hash(payload: Mapping[str, Any], key: str) -> None:
    unsigned = dict(payload)
    stored = unsigned.pop(key)
    if stored != sha256_bytes(canonical_json_bytes(unsigned)):
        raise ValueError(f"{key} self-hash mismatch")


def replay_repair_bank(
    manifest_path: Path,
    *,
    report_path: Path | None = None,
    rerun_motion: bool = True,
) -> dict[str, Any]:
    """Replay every source, plan and rest decision; optionally rerender 75,520 frames."""
    manifest_path = Path(manifest_path).resolve()
    bank_root = manifest_path.parent
    bank = load_schema_json(
        manifest_path,
        maximum_bytes=MAX_BANK_BYTES,
        label="neural rig repair bank",
        schema=BANK_SCHEMA,
    )
    _verify_self_hash(bank, "bank_sha256")
    if bank["source"]["frozen_bridge_source_sha256"] != EXPECTED_BRIDGE_SOURCE_SHA256:
        raise ValueError("repair bank frozen bridge hash mismatch")
    if bank["source"]["repair_source_sha256"] != source_hash():
        raise ValueError("repair bank implementation source hash mismatch")
    generation_path = resolve_artifact_record(
        PROJECT_ROOT,
        bank["source"]["generation_manifest"],
        label="repair generation manifest",
        maximum_bytes=MAX_JSON_BYTES,
    )
    style_path = resolve_artifact_record(
        PROJECT_ROOT,
        bank["source"]["style_manifest"],
        label="repair style manifest",
        maximum_bytes=MAX_JSON_BYTES,
    )
    source = load_repair_source(generation_path, style_path)
    if (
        source.generation_manifest_sha256
        != bank["source"]["generation_manifest"]["sha256"]
        or source.style_manifest_sha256 != bank["source"]["style_manifest"]["sha256"]
        or source.legal_tuple_fingerprint != bank["source"]["legal_tuple_fingerprint"]
    ):
        raise ValueError("repair bank source linkage mismatch")
    if [record["ordinal"] for record in bank["plans"]] != list(
        range(EXPECTED_SAMPLE_COUNT)
    ):
        raise ValueError("repair bank plans are not in canonical ordinal order")

    bindings = []
    identity_results: list[dict[str, Any]] = []
    plans = []
    for sample, record in zip(source.samples, bank["plans"], strict=True):
        if (
            record["sample_id"] != sample.sample_id
            or record["ordinal"] != sample.ordinal
            or record["family"] != sample.family
        ):
            raise ValueError("repair bank plan identity registry mismatch")
        plan_path = resolve_artifact_record(
            bank_root,
            record["artifact"],
            label=f"repair plan {sample.sample_id}",
            maximum_bytes=MAX_PLAN_BYTES,
        )
        plan = load_repair_plan(plan_path)
        exact_plan = compile_repair_plan(source, sample)
        if canonical_json_bytes(plan) != canonical_json_bytes(exact_plan):
            raise ValueError(f"repair plan replay mismatch for {sample.sample_id}")
        if plan["hashes"]["plan_sha256"] != record["plan_sha256"]:
            raise ValueError("repair plan registry hash mismatch")
        binding = bind_repair_plan(source, sample, plan, verify_exact_plan=True)
        if binding.sha256 != record["binding_sha256"]:
            raise ValueError("repair binding registry hash mismatch")
        errors = validate_repaired_binding(binding)
        if errors:
            raise ValueError("replayed binding invalid: " + "; ".join(errors))
        reconstructed = binding.reconstruct_fields()
        rest_exact = all(
            np.array_equal(actual, expected)
            for actual, expected in zip(
                reconstructed,
                (sample.part_owner, sample.material, sample.emission_level),
                strict=True,
            )
        )
        if not rest_exact:
            raise ValueError(f"replayed rest arrays changed for {sample.sample_id}")
        bindings.append(binding)
        plans.append(plan)
        identity_results.append(
            {
                "sample_id": sample.sample_id,
                "ordinal": sample.ordinal,
                "family": sample.family,
                "plan_artifact_exact": True,
                "plan_self_hash_exact": True,
                "exact_source_recompile": True,
                "binding_manifest_exact": True,
                "part_owner_rest_exact": True,
                "material_rest_exact": True,
                "emission_rest_exact": True,
                "raw_fields_hash_exact": binding.raw_fields_sha256
                == sample.raw_fields_sha256,
                "motion_audit_exact": False,
            }
        )

    rest_path = resolve_artifact_record(
        bank_root,
        bank["artifacts"]["rest_audit"],
        label="repair rest audit",
        maximum_bytes=MAX_BANK_BYTES,
    )
    rest = load_rest_audit(rest_path)
    if (
        rest["source"]["generation_manifest_sha256"]
        != source.generation_manifest_sha256
        or rest["source"]["style_manifest_sha256"] != source.style_manifest_sha256
        or rest["source"]["legal_tuple_fingerprint"] != source.legal_tuple_fingerprint
    ):
        raise ValueError("repair rest audit source linkage mismatch")
    expected_rejections = [
        {
            "family": family,
            "family_index": family_index,
            "sample_id": sample_id,
            "category": category,
        }
        for family, family_index, sample_id, category in EXPECTED_REJECTIONS
    ]
    if (
        rest["baseline_v1"]["rejections"] != expected_rejections
        or bank["baseline_v1"] != rest["baseline_v1"]
        or bank["repaired_v2"] != rest["repaired_v2"]
    ):
        raise ValueError("repair census or rejection identity registry mismatch")
    for result, rest_sample, record in zip(
        identity_results, rest["samples"], bank["plans"], strict=True
    ):
        if (
            rest_sample["sample_id"] != result["sample_id"]
            or rest_sample["plan_sha256"] != record["plan_sha256"]
            or rest_sample["binding_sha256"] != record["binding_sha256"]
            or any(value is not True for value in rest_sample["gates"].values())
        ):
            raise ValueError("repair rest audit identity linkage mismatch")

    stress_path = resolve_artifact_record(
        bank_root,
        bank["artifacts"]["motion_stress"],
        label="repair motion stress report",
        maximum_bytes=MAX_STRESS_REPORT_BYTES,
    )
    stress = load_stress_report(stress_path, verify_shards=True)
    expected_stress_source = {
        "generation_manifest_sha256": source.generation_manifest_sha256,
        "style_manifest_sha256": source.style_manifest_sha256,
        "frozen_bridge_source_sha256": EXPECTED_BRIDGE_SOURCE_SHA256,
        "repair_source_sha256": source_hash(),
    }
    if (
        stress["stress_sha256"] != bank["motion_result"]["stress_sha256"]
        or stress["source"] != expected_stress_source
        or stress["sample_count"] != EXPECTED_SAMPLE_COUNT
        or stress["clip_count"] != EXPECTED_CLIP_COUNT
        or stress["frame_count"] != EXPECTED_FRAME_COUNT
    ):
        raise ValueError("repair motion stress linkage mismatch")
    telemetry_path = resolve_artifact_record(
        bank_root,
        bank["artifacts"]["crash_telemetry"],
        label="repair crash telemetry",
        maximum_bytes=MAX_BANK_BYTES,
    )
    telemetry = load_strict_json(
        telemetry_path,
        maximum_bytes=MAX_BANK_BYTES,
        label="repair crash telemetry",
    )
    if (
        telemetry.get("format")
        != "nullvector-neural-rig-repair-stress-telemetry-v1"
        or telemetry.get("cpu_only_environment") is not True
        or telemetry.get("failed_shards") != []
    ):
        raise ValueError("repair crash telemetry contract failed")

    shard_samples: dict[int, Mapping[str, Any]] = {}
    for shard_record in stress["shards"]:
        shard_path = resolve_artifact_record(
            stress_path.parent,
            shard_record,
            label="repair stress shard replay",
            maximum_bytes=MAX_STRESS_SHARD_BYTES,
        )
        shard = load_stress_shard(shard_path)
        for sample_record in shard["samples"]:
            ordinal = int(sample_record["ordinal"])
            if ordinal in shard_samples:
                raise ValueError("repair stress sample was duplicated across shards")
            shard_samples[ordinal] = sample_record
    if set(shard_samples) != set(range(EXPECTED_SAMPLE_COUNT)):
        raise ValueError("repair stress shards do not cover all identities")
    for ordinal, (binding, plan, result) in enumerate(
        zip(bindings, plans, identity_results, strict=True)
    ):
        stored = shard_samples[ordinal]
        if (
            stored["sample_id"] != binding.sample_id
            or stored["plan_sha256"] != plan["hashes"]["plan_sha256"]
            or stored["binding_sha256"] != binding.sha256
            or stored["raw_fields_sha256"] != binding.raw_fields_sha256
        ):
            raise ValueError("repair stress shard identity linkage mismatch")
        if rerun_motion:
            replayed_motion = compile_sample_motion_audit(binding)
            if canonical_json_bytes(replayed_motion) != canonical_json_bytes(
                stored["motion_audit"]
            ):
                raise ValueError(
                    f"exact repair motion replay mismatch for {binding.sample_id}"
                )
            result["motion_audit_exact"] = True

    mode = "exact_motion" if rerun_motion else "metadata_only"
    status = "passed" if rerun_motion else "inspected"
    report_root = (
        Path(report_path).resolve().parent if report_path is not None else bank_root
    )
    if not manifest_path.is_relative_to(report_root):
        # Reports outside the bank root use a simple name/hash record rather
        # than an unsafe upward path; this record is descriptive, while the
        # already-open manifest remains the replay authority.
        bank_record = {
            "path": manifest_path.name,
            "bytes": manifest_path.stat().st_size,
            "sha256": sha256_file(manifest_path),
        }
    else:
        bank_record = artifact_record(manifest_path, report_root)
    report_base: dict[str, Any] = {
        "format": REPLAY_FORMAT,
        "status": status,
        "mode": mode,
        "neural_output": rerun_motion,
        "bank_manifest": bank_record,
        "bank_sha256": bank["bank_sha256"],
        "source": {
            "generation_manifest_sha256": source.generation_manifest_sha256,
            "style_manifest_sha256": source.style_manifest_sha256,
            "legal_tuple_fingerprint": source.legal_tuple_fingerprint,
            "frozen_bridge_source_sha256": EXPECTED_BRIDGE_SOURCE_SHA256,
            "repair_source_sha256": source_hash(),
        },
        "counts": {
            "sample_count": EXPECTED_SAMPLE_COUNT,
            "plan_count": EXPECTED_SAMPLE_COUNT,
            "binding_count": EXPECTED_SAMPLE_COUNT,
            "rest_array_count": EXPECTED_SAMPLE_COUNT * 3,
            "rest_cell_count": EXPECTED_SAMPLE_COUNT * 48 * 48,
            "clip_count": EXPECTED_CLIP_COUNT,
            "frame_count": EXPECTED_FRAME_COUNT,
            "artifact_count_compared": 261,
        },
        "identity_results": identity_results,
        "gates": {
            "bank_schema_valid": True,
            "bank_self_hash_exact": True,
            "source_artifact_records_exact": True,
            "all_80_raw_manifest_and_archive_pairs_exact": True,
            "frozen_bridge_source_hash_exact": True,
            "repair_source_hash_exact": True,
            "all_80_plan_artifacts_exact": True,
            "all_80_plan_self_hashes_exact": True,
            "all_80_plans_exactly_source_recompiled": True,
            "all_80_binding_manifests_exact": True,
            "all_240_rest_arrays_byte_exact": True,
            "frozen_v1_70_10_census_exact": True,
            "exact_10_rejection_identity_registry": True,
            "all_16_stress_shards_exact": True,
            "all_8320_stored_clip_audits_exact": True,
            "all_75520_stored_frame_records_exact": True,
            "all_8320_clip_audits_exactly_rerendered": rerun_motion,
            "all_gates_true": rerun_motion,
        },
    }
    report_base["replay_sha256"] = sha256_bytes(canonical_json_bytes(report_base))
    validate_schema(report_base, REPLAY_SCHEMA)
    if report_path is not None:
        write_canonical_json(
            report_path,
            report_base,
            replace=False,
        )
    return report_base
