from __future__ import annotations

from forge.cellular_ontogeny_sync import DEFAULT_DESTINATION, DEFAULT_SOURCE, project_runtime, validate_runtime


def test_native_projection_is_exact() -> None:
    assert project_runtime(DEFAULT_SOURCE) == project_runtime(DEFAULT_SOURCE)


def test_native_catalog_validates_when_present() -> None:
    if DEFAULT_DESTINATION.is_dir(): assert validate_runtime(DEFAULT_DESTINATION)["passed"]
