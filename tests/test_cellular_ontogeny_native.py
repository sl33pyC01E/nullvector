from __future__ import annotations

import json

from forge.cellular_ontogeny_sync import DEFAULT_DESTINATION, DEFAULT_SOURCE, project_runtime, validate_runtime


REPORT = DEFAULT_SOURCE.parents[2] / "outputs/cellular_ontogeny_godot_report_v2.json"


def test_native_projection_is_exact() -> None:
    assert project_runtime(DEFAULT_SOURCE) == project_runtime(DEFAULT_SOURCE)


def test_native_catalog_validates_when_present() -> None:
    if DEFAULT_DESTINATION.is_dir(): assert validate_runtime(DEFAULT_DESTINATION)["passed"]


def test_native_smoke_uses_current_organ_motion_and_ecology_chain() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["format"] == "nullvector-cellular-ontogeny-godot-smoke-v2"
    assert report["passed"] is True and report["program_count"] == 45
