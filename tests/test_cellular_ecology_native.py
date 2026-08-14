from __future__ import annotations

import json

from forge.cellular_ecology_sync import DEFAULT_DESTINATION, DEFAULT_SOURCE, project_runtime, validate_runtime


ROOT = DEFAULT_SOURCE.parents[2]


def test_native_projection_is_exact() -> None:
    assert project_runtime(DEFAULT_SOURCE) == project_runtime(DEFAULT_SOURCE)


def test_native_catalog_validates_when_present() -> None:
    if not DEFAULT_DESTINATION.is_dir(): return
    report = validate_runtime(DEFAULT_DESTINATION)
    assert report["passed"] and report["map_count"] == 6 and report["resource_node_count"] == 120


def test_ecology_scene_uses_current_motion_physiology_and_trauma_bundles() -> None:
    scene = (ROOT / "game/CellularEcologyLab.tscn").read_text(encoding="utf-8"); script = (ROOT / "game/scripts/cellular_ecology_lab.gd").read_text(encoding="utf-8")
    for path in ("cellular_motion/v4", "cellular_physiology/v3", "cellular_trauma/v1", "cellular_ecology/v1"):
        assert path in scene
    for feature in (
        "_resource_affinity", "_is_resource_cell", "_step_ecology_motility",
        "_step_ecology_behavior", "_seed_ecology_population",
        "family_suitability_u8", "ecology_motive_impulse",
    ):
        assert feature in script


def test_godot_ecology_smoke_proves_differentiated_metabolism_and_resource_seeking() -> None:
    report = json.loads((ROOT / "outputs/cellular_ecology_godot_report_v3.json").read_text(encoding="utf-8"))
    assert report["format"] == "nullvector-cellular-ecology-godot-smoke-v3" and report["passed"] is True
    assert report["differentiated_metabolism_verified"] is True
    assert report["autonomous_motion_verified"] is True
    assert report["action_cycle_verified"] is True
    assert report["organ_capacity_motion_gate_verified"] is True
    assert report["organism_count"] == 5 and report["family_census"] == {str(index): 1 for index in range(5)}
    assert len(report["motion_census"]) >= 3 and len(report["behavior_census"]) == 5
    assert report["map_count"] == 6 and report["resource_node_count"] == 120
    for key in ("ecology_bundle_id", "motion_bundle_id", "physiology_bundle_id", "trauma_bundle_id", "organism_bundle_id"):
        assert len(report[key]) == 64
