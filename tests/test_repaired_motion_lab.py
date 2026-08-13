from __future__ import annotations

import json
from pathlib import Path

from forge.config import PROJECT_ROOT
from forge.repaired_motion_lab_sync import (
    DEFAULT_DESTINATION,
    FACINGS,
    LAYERS,
    MOTIONS,
    project_runtime,
    validate_runtime,
)


def test_all_80_runtime_projection_is_balanced_and_complete() -> None:
    inline, sources = project_runtime()
    catalog = json.loads(inline["catalog.json"])
    assert catalog["status"] == "ready"
    assert catalog["neural_output"] is True
    assert catalog["counts"] == {
        "identity_count": 80,
        "family_count": 5,
        "motion_count": 13,
        "facing_count": 8,
        "clip_count": 8320,
        "frame_count": 75520,
        "atlas_count": 560,
    }
    assert set(catalog["family_counts"].values()) == {16}
    assert catalog["layers"] == list(LAYERS)
    assert catalog["motions"] == list(MOTIONS)
    assert catalog["facings"] == list(FACINGS)
    assert len(sources) == 560
    assert len(catalog["identities"]) == 80
    assert [identity["ordinal"] for identity in catalog["identities"]] == list(range(80))
    assert all(len(identity["clips"]) == 104 for identity in catalog["identities"])
    assert all(len(identity["atlases"]) == 7 for identity in catalog["identities"])


def test_projection_repeats_exactly_without_mutating_sources() -> None:
    first_inline, first_sources = project_runtime()
    second_inline, second_sources = project_runtime()
    assert first_inline == second_inline
    assert first_sources == second_sources


def test_checked_in_runtime_projection_validates_exhaustively() -> None:
    report = validate_runtime(DEFAULT_DESTINATION)
    assert report["passed"] is True
    assert report["identity_count"] == 80
    assert report["atlas_count"] == 560
    assert report["clip_count"] == 8320
    assert report["frame_count"] == 75520


def test_native_lab_declares_exact_runtime_and_loop_contracts() -> None:
    scene = (PROJECT_ROOT / "game/RepairedMotionLab.tscn").read_text(encoding="utf-8")
    script = (PROJECT_ROOT / "game/scripts/repaired_motion_lab.gd").read_text(encoding="utf-8")
    assert "res://scripts/repaired_motion_lab.gd" in scene
    assert 'const CATALOG_PATH := "res://generated/repaired_motion_lab/v1/catalog.json"' in script
    assert "stored - 1 if clip.get(\"loop\", false) else stored" in script
    assert "atlas_count != 560" in script
    assert "clip_count != 8320" in script
    assert "frame_count != 75520" in script


def test_latest_godot_smoke_proves_every_atlas_and_frame_region() -> None:
    report_path = PROJECT_ROOT / "outputs/repaired_motion_lab_godot_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["identity_count"] == 80
    assert report["family_counts"] == {
        "humanoid": 16,
        "animalian": 16,
        "plantlike": 16,
        "anomaly": 16,
        "machine": 16,
    }
    assert report["atlas_count"] == 560
    assert report["clip_count"] == 8320
    assert report["frame_count"] == 75520
    assert report["atlas_regions_checked"] == 75520
    assert report["python_runtime_required"] is False
    assert report["errors"] == []
