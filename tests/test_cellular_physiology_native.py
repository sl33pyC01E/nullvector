from __future__ import annotations

import json
from pathlib import Path

from forge.cellular_physiology_sync import project_runtime, validate_runtime


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/cellular_physiology_v2/cellular_physiology_manifest.json"
RUNTIME = ROOT / "game/generated/cellular_physiology/v5"
SCENE = ROOT / "game/CellularMotionLab.tscn"
SCRIPT = ROOT / "game/scripts/cellular_motion_lab.gd"


def test_native_physiology_projection_is_repeatable_and_hash_closed() -> None:
    assert project_runtime(SOURCE) == project_runtime(SOURCE)
    validation = validate_runtime(RUNTIME)
    assert (validation["passed"], validation["identity_count"], validation["system_count"], validation["cell_count"]) == (True, 45, 8, 25_668)
    assert validation["bytes"] > 4_000_000 and len(validation["bundle_id"]) == 64


def test_motion_lab_loads_and_applies_connected_physiology() -> None:
    scene = SCENE.read_text(encoding="utf-8"); script = SCRIPT.read_text(encoding="utf-8")
    assert 'physiology_catalog_path = "res://generated/cellular_physiology/v5/catalog.json"' in scene
    for feature in (
        "_compute_physiology_capacities", "_physiology_reachable", "_prepare_physiology",
        "_advance_physiology", "physiology_oxygen", "physiology_base_digestion",
        "physiology_network_reachable", "motion_chain_depth", "graded local physiology",
        "local_circulation", "local_immune",
    ):
        assert feature in script


def test_godot_smoke_proves_brain_damage_cascade_and_runtime_census() -> None:
    report = json.loads((ROOT / "outputs/cellular_physiology_motion_godot_report_v6.json").read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert (report["physiology_identity_count"], report["physiology_system_count"]) == (45, 8)
    assert report["physiology_core_damage_verified"] is True
    assert report["all_system_core_failures_verified"] is True
    assert report["member_routing_verified"] is True
    assert report["graded_local_delivery_verified"] is True
    assert report["progressive_chain_verified"] is True
    assert report["population_after_reproduction"] == 2
