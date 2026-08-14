from __future__ import annotations

from forge.cellular_ecology_sync import DEFAULT_DESTINATION, DEFAULT_SOURCE, project_runtime, validate_runtime


def test_native_projection_is_exact() -> None:
    assert project_runtime(DEFAULT_SOURCE) == project_runtime(DEFAULT_SOURCE)


def test_native_catalog_validates_when_present() -> None:
    if not DEFAULT_DESTINATION.is_dir(): return
    report = validate_runtime(DEFAULT_DESTINATION)
    assert report["passed"] and report["map_count"] == 6 and report["resource_node_count"] == 120
