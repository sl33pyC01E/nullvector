from __future__ import annotations

import json
from pathlib import Path

from forge.cellular_physiology_sync import project_runtime, validate_runtime


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/cellular_physiology_v3/cellular_physiology_manifest.json"
RUNTIME = ROOT / "game/generated/cellular_physiology/v10"
SCENE = ROOT / "game/CellularMotionLab.tscn"
SCRIPT = ROOT / "game/scripts/cellular_motion_lab.gd"


def test_native_physiology_projection_is_repeatable_and_hash_closed() -> None:
    assert project_runtime(SOURCE) == project_runtime(SOURCE)
    validation = validate_runtime(RUNTIME)
    assert (validation["passed"], validation["identity_count"], validation["system_count"], validation["cell_count"]) == (True, 45, 8, 25_668)
    assert validation["bytes"] > 4_000_000 and len(validation["bundle_id"]) == 64


def test_motion_lab_loads_and_applies_connected_physiology() -> None:
    scene = SCENE.read_text(encoding="utf-8"); script = SCRIPT.read_text(encoding="utf-8")
    assert 'physiology_catalog_path = "res://generated/cellular_physiology/v10/catalog.json"' in scene
    for feature in (
        "_compute_physiology_capacities", "_physiology_reachable", "_prepare_physiology",
        "_advance_physiology", "physiology_oxygen", "physiology_base_digestion",
        "physiology_network_reachable", "motion_chain_depth", "graded local physiology",
        "local_circulation", "local_immune",
        "_compute_homeostasis_capacities", "physiology_functional_capacities",
        "_diagnostic_homeostasis_matrix", "homeostasis_incapacitated",
    ):
        assert feature in script


def test_godot_smoke_proves_all_identity_organ_cascades_and_runtime_census() -> None:
    report = json.loads((ROOT / "outputs/cellular_physiology_motion_godot_report_v8.json").read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["format"] == "nullvector-cellular-motion-godot-smoke-v8"
    assert (report["physiology_identity_count"], report["physiology_system_count"]) == (45, 8)
    assert report["physiology_core_damage_verified"] is True
    assert report["all_system_core_failures_verified"] is True
    assert report["member_routing_verified"] is True
    assert report["graded_local_delivery_verified"] is True
    assert report["local_perfusion_verified"] is True
    assert report["progressive_chain_verified"] is True
    assert report["reserve_aware_homeostasis_verified"] is True
    homeostasis = report["homeostasis_matrix"]
    assert homeostasis["passed"] is True
    assert homeostasis["heart"]["initial"]["circulation"] == 0.0
    assert homeostasis["respiratory"]["initial"]["consciousness"] > 0.5
    assert homeostasis["respiratory"]["time_to_incapacitation"] > 0.0
    assert homeostasis["respiratory"]["final_oxygen"] < 0.14
    assert homeostasis["digestive"]["initial"]["reproduction"] == 0.0
    assert homeostasis["digestive"]["initial"]["circulation"] > 0.5
    assert homeostasis["brain"]["initial"]["consciousness"] == 0.0
    assert homeostasis["brain"]["initial"]["circulation"] > 0.5
    assert report["population_after_reproduction"] == 2
    matrix = report["full_identity_failure_matrix"]
    assert matrix["passed"] is True
    assert matrix["family_census_valid"] is True
    assert (matrix["identity_count"], matrix["core_failure_case_count"], matrix["cascade_signature_count"]) == (45, 360, 180)
    assert matrix["family_counts"] == {"humanoid": 11, "animalian": 10, "plantlike": 9, "anomaly": 8, "machine": 7}
    assert matrix["minimum_retained_circulation"] > 0.5
