from __future__ import annotations

import json
from pathlib import Path

from forge.cellular_trauma_sync import project_runtime, validate_runtime


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/cellular_trauma_v2/cellular_trauma_manifest.json"
RUNTIME = ROOT / "game/generated/cellular_trauma/v2"
SCENE = ROOT / "game/CellularMotionLab.tscn"
SCRIPT = ROOT / "game/scripts/cellular_motion_lab.gd"
REPORT = ROOT / "outputs/cellular_physiology_motion_godot_report_v6.json"


def test_native_trauma_projection_is_repeatable_and_hash_closed() -> None:
    assert project_runtime(SOURCE) == project_runtime(SOURCE)
    validation = validate_runtime(RUNTIME)
    assert (validation["passed"], validation["identity_count"], validation["total_cells"], validation["total_bonds"]) == (True, 45, 25_668, 85_357)
    assert validation["bytes"] > 3_500_000 and len(validation["bundle_id"]) == 64


def test_motion_lab_loads_clots_scars_magnetism_and_fragment_fates() -> None:
    scene = SCENE.read_text(encoding="utf-8"); script = SCRIPT.read_text(encoding="utf-8")
    assert 'trauma_catalog_path = "res://generated/cellular_trauma/v2/catalog.json"' in scene
    for feature in ("_step_trauma_magnetism", "_step_trauma_components", "trauma_clot", "trauma_scar", "trauma_fragment_fate", "trauma_reconnections", "trauma_polyps"):
        assert feature in script


def test_godot_smoke_proves_reconnection_scar_and_plant_polyp() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["format"] == "nullvector-cellular-motion-godot-smoke-v6" and report["passed"] is True
    assert report["trauma_identity_count"] == 45 and len(report["trauma_bundle_id"]) == 64
    assert report["trauma_reconnection_verified"] is True and report["plant_polyp_verified"] is True
    assert report["physiology_core_damage_verified"] is True and report["python_runtime_required"] is False
