from __future__ import annotations

import json
from pathlib import Path

import pytest

from forge.morphology_diversity import (
    DEFAULT_OUTPUT,
    FEATURE_NAMES,
    audit_morphology_diversity,
    build_report,
    validate_report,
)


@pytest.fixture(scope="module")
def report() -> dict:
    return audit_morphology_diversity()


def test_shape_only_audit_proves_family_and_chassis_diversity(report: dict) -> None:
    assert report["status"] == "ready"
    assert len(FEATURE_NAMES) == 154
    assert report["aggregate"]["sample_count"] == 80
    assert report["aggregate"]["unique_coarse_chassis_count"] >= 72
    assert report["classification"]["family"]["leave_one_out_accuracy"] >= 0.90
    assert all(row["recall"] >= 0.75 for row in report["classification"]["family"]["per_class"])


def test_subtype_role_and_symmetry_are_reported_without_false_hard_claims(report: dict) -> None:
    assert report["classification"]["subtype"]["class_count"] == 20
    assert report["classification"]["role"]["class_count"] == 8
    assert report["interpretation"]["symmetry_is_hard_requirement"] is False
    assert len(report["aggregate"]["per_family"]) == 5
    assert all(row["horizontal_symmetry"]["mean"] > 0.5 for row in report["aggregate"]["per_family"])


def test_publication_is_immutable_canonical_and_exactly_replayable(tmp_path: Path) -> None:
    path = tmp_path / "morphology_diversity.json"
    built = build_report(path)
    assert built["passed"] is True
    assert validate_report(path) == built
    payload = json.loads(path.read_bytes())
    payload["records"][0]["metrics"]["horizontal_symmetry"] = 0
    path.write_text(json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        validate_report(path)


def test_published_authority_replays_when_present() -> None:
    if DEFAULT_OUTPUT.is_file():
        assert validate_report(DEFAULT_OUTPUT)["passed"] is True
