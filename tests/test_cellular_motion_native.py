from __future__ import annotations

import hashlib
import json
from pathlib import Path

from forge.cellular_motion_sync import project_runtime, validate_runtime


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/cellular_motion_v2/cellular_motion_manifest.json"
RUNTIME = ROOT / "game/generated/cellular_motion/v11"
SCENE = ROOT / "game/CellularMotionLab.tscn"
SCRIPT = ROOT / "game/scripts/cellular_motion_lab.gd"


def test_motion_native_projection_is_repeatable_and_closed() -> None:
    assert project_runtime(SOURCE) == project_runtime(SOURCE)
    validation = validate_runtime(RUNTIME)
    assert (validation["passed"], validation["identity_count"], validation["clip_count"], validation["frame_count"]) == (True, 45, 520, 4720)
    assert validation["bytes"] > 2_000_000 and len(validation["bundle_id"]) == 64


def test_native_scene_uses_live_organ_drivers_not_sprite_frames() -> None:
    scene = SCENE.read_text(encoding="utf-8"); script = SCRIPT.read_text(encoding="utf-8")
    assert 'motion_catalog_path = "res://generated/cellular_motion/v11/motion_catalog.json"' in scene
    assert "expected_species_count = 45" in scene
    for feature in (
        "_apply_motion_force", "_channel_driver", "_neural_reachable_cells",
        "_channel_integrities", "_current_frame", "_set_organism_motion",
        "motion_last_event_frame", "_diagnostic_system_core_failures",
        "_draw_physiology_network", "SYSTEM NETWORK", "_select_system",
        "_compute_homeostasis_capacities", "physiology_functional_capacities",
        "_diagnostic_homeostasis_matrix", "_diagnostic_full_identity_actuation_matrix",
        "_diagnostic_full_identity_connection_matrix", "_diagnostic_connection_sever",
    ):
        assert feature in script
    assert "AtlasTexture" not in script and "Sprite2D" not in script


def test_native_smoke_exhausts_motion_programs_and_actuates_damageable_body() -> None:
    report = json.loads((ROOT / "outputs/cellular_physiology_motion_godot_report_v8.json").read_text(encoding="utf-8"))
    assert report["format"] == "nullvector-cellular-motion-godot-smoke-v8"
    assert report["passed"] is True
    assert (report["identity_count"], report["clip_count"], report["frame_count"]) == (45, 520, 4720)
    assert report["mapped_organs"] == 748 and report["event_count"] == 85
    assert report["actuation_velocity"] > 100.0
    assert report["damage_killed"] > 0 and report["damage_bonds"] > 0
    assert report["all_system_core_failures_verified"] is True
    assert report["system_view_verified"] is True
    assert report["member_routing_verified"] is True
    assert report["graded_local_delivery_verified"] is True
    assert report["local_perfusion_verified"] is True
    assert report["progressive_chain_verified"] is True
    assert report["full_identity_failure_matrix"]["passed"] is True
    assert report["reserve_aware_homeostasis_verified"] is True
    matrix = report["homeostasis_matrix"]
    assert matrix["passed"] is True
    assert matrix["respiratory"]["initial"]["consciousness"] > 0.5
    assert matrix["respiratory"]["time_to_incapacitation"] > 0.0
    assert matrix["respiratory"]["final_oxygen"] < 0.14
    assert matrix["brain"]["initial"]["consciousness"] == 0.0
    assert matrix["digestive"]["initial"]["reproduction"] == 0.0
    actuation = report["full_identity_actuation_matrix"]
    assert actuation["passed"] is True
    assert (actuation["identity_count"], actuation["case_count"], actuation["failures"]) == (45, 225, [])
    assert actuation["family_counts"] == {
        "humanoid": 11, "animalian": 10, "plantlike": 9,
        "anomaly": 8, "machine": 7,
    }
    assert all(value > 0.001 for value in actuation["minimum_channel_velocity"].values())
    connections = report["full_identity_connection_matrix"]
    assert connections["passed"] is True
    assert (connections["identity_count"], connections["case_count"], connections["failures"]) == (45, 315, [])
    assert connections["system_counts"] == {
        "circulation": 45, "respiration": 45, "digestion": 45,
        "neural": 45, "sensory": 45, "locomotion": 45,
        "reproduction": 45,
    }
    assert connections["minimum_capacity_drop"] > 0.0001
    assert "core-only humoral" in connections["immune_model"]
    assert set(report["system_core_failures"]) == {
        "circulation", "respiration", "digestion", "neural", "sensory",
        "locomotion", "reproduction", "immune",
    }
    assert all(value["core_cells"] > 0 and value["remaining_capacity"] == 0 for value in report["system_core_failures"].values())
    assert report["population_after_reproduction"] == 2 and report["python_runtime_required"] is False


def test_arena_remains_the_untouched_main_scene() -> None:
    assert 'run/main_scene="res://Arena.tscn"' in (ROOT / "game/project.godot").read_text(encoding="utf-8")
    assert hashlib.sha256((ROOT / "game/Arena.tscn").read_bytes()).hexdigest() == "c5f8b961297b43f683d40dff831cb576d89539c222f0a5a9abab3d29a1f67490"
