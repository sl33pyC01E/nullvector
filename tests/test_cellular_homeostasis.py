from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from forge.cellular_homeostasis import DEFAULT_OUTPUT, build_report, validate_report, write_report


ROOT = Path(__file__).resolve().parents[1]


def test_all_organs_have_distinct_structural_and_functional_failure_signatures() -> None:
    report = build_report()
    assert report["status"] == "ready"
    assert report["identity_count"] == 45 and report["core_failure_case_count"] == 360
    assert all(report["gates"].values())
    for record in report["core_failure_matrix"]:
        assert all(failure["baseline_own_capacity"] > 0.99 for failure in record["failures"].values())
        assert all(failure["own_capacity"] == 0.0 for failure in record["failures"].values())


def test_reserves_make_lung_gut_brain_and_heart_lesions_temporally_distinct() -> None:
    report = build_report()
    for family in report["family_scenarios"]:
        cases = family["cases"]
        assert cases["brain"]["initial"]["incapacitated"] is True
        assert cases["brain"]["initial"]["circulation"] > 0.5
        assert cases["respiratory"]["initial"]["consciousness"] > 0.5
        assert cases["respiratory"]["time_to_incapacitation"] > 0.0
        assert cases["digestive"]["initial"]["reproduction"] == 0.0
        assert cases["digestive"]["final"]["energy"] < cases["digestive"]["initial"]["energy"]
        assert cases["heart"]["time_to_death"] is not None


def test_report_is_canonical_exactly_replayable_and_rejects_tamper(tmp_path: Path) -> None:
    destination = tmp_path / "homeostasis.json"
    result = write_report(destination)
    assert result["passed"] is True and result["dynamic_scenario_count"] == 20
    assert validate_report(destination) == result
    payload = json.loads(destination.read_text(encoding="utf-8"))
    payload["family_scenarios"][0]["cases"]["brain"]["initial"]["consciousness"] = 0.5
    destination.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    with pytest.raises(ValueError):
        validate_report(destination)


def test_published_report_replays_when_present() -> None:
    if DEFAULT_OUTPUT.is_file():
        assert validate_report(DEFAULT_OUTPUT)["passed"] is True
