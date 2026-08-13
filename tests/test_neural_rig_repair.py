from __future__ import annotations

from collections import Counter
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from forge.neural_rig_bridge.model import DRIVER_INDEX, DRIVER_NAMES
from forge.neural_rig_repair.binding import bind_repair_plan, validate_repaired_binding
from forge.neural_rig_repair.compiler import BANK_FILENAME, VERIFICATION_FILENAME
from forge.neural_rig_repair.constants import (
    BANK_SCHEMA,
    DEFAULT_GENERATION_MANIFEST,
    DEFAULT_OUTPUT,
    DEFAULT_STYLE_MANIFEST,
    EXPECTED_BRIDGE_SOURCE_SHA256,
    EXPECTED_REJECTION_CATEGORIES,
    EXPECTED_REJECTIONS,
    PLAN_SCHEMA,
    PROJECT_ROOT,
    REPAIR_MIN_DRIVER_PIXELS,
    REPLAY_SCHEMA,
)
from forge.neural_rig_repair.hashing import canonical_json_bytes, source_hash
from forge.neural_rig_repair.motion import (
    MAX_POSED_ANCHOR_SUPPORT_DISTANCE,
    _minimum_pixel_support_distance,
    _project_physical_driver_points,
    compile_motion_clip_audit,
    compile_sample_motion_audit,
)
from forge.neural_rig_repair.planner import compile_repair_plan, load_repair_plan
from forge.neural_rig_repair.replay import replay_repair_bank
from forge.neural_rig_repair.schema import (
    load_strict_json,
    resolve_artifact_record,
    validate_schema,
)
from forge.neural_rig_repair.source import load_repair_source
from forge.neural_rig_repair.stress import (
    _exit_telemetry,
    load_stress_report,
    load_stress_shard,
)


@pytest.fixture(scope="session")
def repair_matrix():
    source = load_repair_source(DEFAULT_GENERATION_MANIFEST, DEFAULT_STYLE_MANIFEST)
    plans = [compile_repair_plan(source, sample) for sample in source.samples]
    bindings = [
        bind_repair_plan(source, sample, plan)
        for sample, plan in zip(source.samples, plans, strict=True)
    ]
    return source, plans, bindings


def _by_id(repair_matrix, sample_id: str):
    source, plans, bindings = repair_matrix
    ordinal = next(
        sample.ordinal for sample in source.samples if sample.sample_id == sample_id
    )
    return source.samples[ordinal], plans[ordinal], bindings[ordinal]


def test_strict_source_loader_binds_exact_authoritative_80(repair_matrix):
    source, plans, bindings = repair_matrix
    assert len(source.samples) == len(plans) == len(bindings) == 80
    assert source.generation_manifest_path == DEFAULT_GENERATION_MANIFEST.resolve()
    assert source.style_manifest_path == DEFAULT_STYLE_MANIFEST.resolve()
    assert len(source_hash()) == 64
    for sample, binding in zip(source.samples, bindings, strict=True):
        assert sample.raw_fields_sha256 == sample.compiled_fields_sha256
        assert not validate_repaired_binding(binding)
        assert all(
            np.array_equal(actual, expected)
            for actual, expected in zip(
                binding.reconstruct_fields(),
                (sample.part_owner, sample.material, sample.emission_level),
                strict=True,
            )
        )
        assert not sample.part_owner.flags.writeable
        assert not binding.driver_index.flags.writeable
        assert min(
            int((binding.driver_index == DRIVER_INDEX[driver]).sum())
            for driver in DRIVER_NAMES
        ) >= REPAIR_MIN_DRIVER_PIXELS


def test_frozen_bridge_census_and_exact_rejection_identities(repair_matrix):
    source, plans, _bindings = repair_matrix
    statuses = Counter(plan["baseline_v1"]["status"] for plan in plans)
    categories: Counter[str] = Counter()
    identities = []
    for sample, plan in zip(source.samples, plans, strict=True):
        categories.update(plan["baseline_v1"]["categories"])
        for category in plan["baseline_v1"]["categories"]:
            identities.append((sample.family, sample.ordinal % 16, sample.sample_id, category))
    assert statuses == Counter(accepted=70, rejected=10)
    assert dict(categories) == EXPECTED_REJECTION_CATEGORIES
    assert identities == list(EXPECTED_REJECTIONS)
    assert all(
        plan["source"]["frozen_bridge_source_sha256"]
        == EXPECTED_BRIDGE_SOURCE_SHA256
        for plan in plans
    )


def test_all_plans_are_strict_schema_valid_and_self_hashed(repair_matrix, tmp_path):
    _source, plans, _bindings = repair_matrix
    for plan in plans:
        validate_schema(plan, PLAN_SCHEMA)
    path = tmp_path / "plan.json"
    path.write_bytes(canonical_json_bytes(plans[0]))
    assert load_repair_plan(path) == plans[0]

    tampered = copy.deepcopy(plans[0])
    tampered["expected"]["minimum_driver_pixels"] = 13
    path.write_bytes(canonical_json_bytes(tampered))
    with pytest.raises(ValueError, match="schema|Manifest"):
        load_repair_plan(path)


def test_strict_json_and_artifact_loader_reject_tamper(tmp_path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"a": 1, "a": 2}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate key"):
        load_strict_json(duplicate, maximum_bytes=100, label="duplicate")

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"a": NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        load_strict_json(nonfinite, maximum_bytes=100, label="nonfinite")

    regular = tmp_path / "artifact.bin"
    regular.write_bytes(b"bound")
    traversal = {"path": "../artifact.bin", "bytes": 5, "sha256": "0" * 64}
    with pytest.raises(ValueError, match="unsafe"):
        resolve_artifact_record(
            tmp_path, traversal, label="traversal", maximum_bytes=100
        )


def test_exact_logical_repairs_cover_each_rejection_class(repair_matrix):
    _sample, anchor_h, _binding = _by_id(repair_matrix, "0012_f0_s02_r6_v00")
    appendage_tip = next(
        anchor
        for anchor in anchor_h["repair"]["anchors"]
        if anchor["kind"] == "socket" and anchor["name"] == "appendage_tip"
    )
    assert appendage_tip["source_point"] == [12, 21]
    assert appendage_tip["point"] == [13, 21]
    assert appendage_tip["displacement"] == 1.0

    _sample, plant, _binding = _by_id(repair_matrix, "0047_f2_s11_r7_v01")
    assert plant["expected"]["physical_component_count"] == 2
    assert plant["expected"]["logical_link_count"] == 1
    assert plant["repair"]["logical_links"][0]["pixels_inserted"] == 0

    assert _by_id(repair_matrix, "0055_f3_s15_r3_v01")[1]["repair"][
        "required_owner_fallbacks"
    ] == ["head"]
    assert _by_id(repair_matrix, "0056_f3_s12_r4_v00")[1]["repair"][
        "required_owner_fallbacks"
    ] == ["body"]

    for sample_id, unsafe in (
        ("0067_f4_s17_r1_v01", 3),
        ("0072_f4_s16_r4_v00", 3),
        ("0079_f4_s19_r7_v01", 5),
    ):
        margin = _by_id(repair_matrix, sample_id)[1]["repair"]["rest_margin"]
        assert margin == {
            "unsafe_foreground_pixels": unsafe,
            "unsafe_physical_pixels": 0,
            "unsafe_aura_pixels": unsafe,
            "physical_margin_clear": True,
            "aura_deferred_to_motion_envelope": True,
        }


def test_motion_envelope_keeps_neural_identity_and_local_anchor_support(repair_matrix):
    sample, _plan, binding = _by_id(repair_matrix, "0076_f4_s18_r6_v00")
    before = (
        sample.part_owner.tobytes(),
        sample.material.tobytes(),
        sample.emission_level.tobytes(),
    )
    clip = compile_motion_clip_audit(binding, "attack", "north")
    assert clip["motion_strength"] > 0.0
    assert clip["frame_count"] == 8
    assert clip["metrics"]["maximum_anchor_support_distance"] <= (
        MAX_POSED_ANCHOR_SUPPORT_DISTANCE
    )
    assert all(clip["gates"].values())
    assert before == (
        sample.part_owner.tobytes(),
        sample.material.tobytes(),
        sample.emission_level.tobytes(),
    )


def test_anchor_support_distance_uses_rendered_pixel_footprints():
    points = np.asarray([[20, 24]], dtype=np.int64)
    target = np.asarray([23.697955842116002, 23.353174478066997])
    center_distance = float(
        np.sqrt(
            (points[0, 1] - target[0]) ** 2
            + (points[0, 0] - target[1]) ** 2
        )
    )
    footprint_distance = _minimum_pixel_support_distance(points, target)
    assert center_distance > MAX_POSED_ANCHOR_SUPPORT_DISTANCE
    assert footprint_distance == pytest.approx(2.8531744780669968)
    assert footprint_distance < MAX_POSED_ANCHOR_SUPPORT_DISTANCE


def test_anchor_support_projection_is_precomposite_and_physical(repair_matrix):
    _sample, _plan, binding = _by_id(repair_matrix, "0041_f2_s08_r4_v01")
    matrices = {driver: np.eye(3, dtype=np.float64) for driver in DRIVER_NAMES}
    projected = _project_physical_driver_points(binding, matrices)
    for anchor in (*binding.joints.values(), *binding.sockets.values()):
        distance = _minimum_pixel_support_distance(
            projected[anchor.driver], np.asarray(anchor.support_point, dtype=np.float64)
        )
        assert distance == 0.0
        support = projected[anchor.driver]
        assert np.all(
            binding.part_owner[support[:, 0], support[:, 1]] != 16
        )


def test_diagonal_facing_appendage_root_uses_bounded_raster_support(repair_matrix):
    sample, _plan, binding = _by_id(repair_matrix, "0004_f0_s02_r2_v00")
    before = (
        sample.part_owner.tobytes(),
        sample.material.tobytes(),
        sample.emission_level.tobytes(),
    )
    clip = compile_motion_clip_audit(binding, "idle_breathe", "northeast")
    assert clip["motion_strength"] > 0.0
    assert 2.8 < clip["metrics"]["maximum_anchor_support_distance"] <= (
        MAX_POSED_ANCHOR_SUPPORT_DISTANCE
    )
    assert all(clip["gates"].values())
    assert before == (
        sample.part_owner.tobytes(),
        sample.material.tobytes(),
        sample.emission_level.tobytes(),
    )


def test_complete_104_clip_matrix_for_difficult_machine(repair_matrix):
    _sample, _plan, binding = _by_id(repair_matrix, "0076_f4_s18_r6_v00")
    audit = compile_sample_motion_audit(binding)
    assert audit["clip_count"] == 104
    assert audit["frame_count"] == 944
    assert sum(audit["strength_histogram"].values()) == 104
    assert min(float(value) for value in audit["strength_histogram"]) > 0.0
    assert all(audit["gates"].values())


@pytest.mark.parametrize(
    ("return_code", "unsigned", "category"),
    [
        (0, 0, "success"),
        (3221225477, 0xC0000005, "access_violation"),
        (3221226505, 0xC0000409, "stack_buffer_overrun"),
        (3221225725, 0xC00000FD, "stack_overflow"),
        (None, None, "timeout"),
    ],
)
def test_windows_native_crash_telemetry_classification(
    return_code, unsigned, category
):
    assert _exit_telemetry(return_code) == (unsigned, category)


def _require_sealed_bank() -> Path:
    bank = DEFAULT_OUTPUT / BANK_FILENAME
    if not bank.is_file():
        pytest.skip("sealed repair bank is not present in this checkout")
    return bank


def test_sealed_bank_metadata_replay_and_strict_tamper(tmp_path):
    bank = _require_sealed_bank()
    report = replay_repair_bank(bank, rerun_motion=False)
    assert report["status"] == "inspected"
    assert report["counts"] == {
        "sample_count": 80,
        "plan_count": 80,
        "binding_count": 80,
        "rest_array_count": 240,
        "rest_cell_count": 184320,
        "clip_count": 8320,
        "frame_count": 75520,
        "artifact_count_compared": 261,
    }
    validate_schema(report, REPLAY_SCHEMA)

    payload = json.loads(bank.read_text(encoding="utf-8"))
    payload["plans"][0]["plan_sha256"] = "0" * 64
    tampered = tmp_path / BANK_FILENAME
    tampered.write_bytes(canonical_json_bytes(payload))
    with pytest.raises(ValueError, match="self-hash"):
        replay_repair_bank(tampered, rerun_motion=False)


def test_sealed_stress_shard_and_replay_schema_reject_tamper(tmp_path):
    bank_path = _require_sealed_bank()
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    stress_path = resolve_artifact_record(
        bank_path.parent,
        bank["artifacts"]["motion_stress"],
        label="test stress",
        maximum_bytes=2 * 1024 * 1024,
    )
    stress = load_stress_report(stress_path, verify_shards=True)
    shard_path = resolve_artifact_record(
        stress_path.parent,
        stress["shards"][0],
        label="test shard",
        maximum_bytes=16 * 1024 * 1024,
    )
    shard = load_stress_shard(shard_path)
    shard["samples"][0]["motion_audit"]["clips"][0]["frames"][0][
        "frame_sha256"
    ] = "0" * 64
    tampered_shard = tmp_path / "stress_shard_00.json"
    tampered_shard.write_bytes(canonical_json_bytes(shard))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_stress_shard(tampered_shard)

    verification = DEFAULT_OUTPUT / VERIFICATION_FILENAME
    if verification.is_file():
        replay = json.loads(verification.read_text(encoding="utf-8"))
        validate_schema(replay, REPLAY_SCHEMA)
        replay["identity_results"][0]["motion_audit_exact"] = False
        with pytest.raises(ValueError, match="Manifest failed"):
            validate_schema(replay, REPLAY_SCHEMA)


def test_bank_schema_rejects_unknown_properties():
    bank = _require_sealed_bank()
    payload = json.loads(bank.read_text(encoding="utf-8"))
    validate_schema(payload, BANK_SCHEMA)
    payload["unexpected"] = True
    with pytest.raises(ValueError, match="Additional properties"):
        validate_schema(payload, BANK_SCHEMA)
