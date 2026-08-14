from __future__ import annotations

import json
from pathlib import Path

import pytest

from forge.cellular_motion_quality import audit_motion_bank, validate_report


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/cellular_motion_v2/cellular_motion_manifest.json"
REPORT = ROOT / "outputs/cellular_motion_quality_v1/motion_quality_report.json"


def test_every_family_motion_has_semantic_noncollapse_evidence() -> None:
    report = audit_motion_bank(SOURCE)
    assert report["status"] == "passed" and report["record_count"] == 65
    assert all(report["gates"].values())
    assert report["aggregate"]["minimum_action_appendage_excursion"] >= 0.5
    assert report["aggregate"]["minimum_locomotor_excursion"] >= 1.0
    assert report["aggregate"]["maximum_locomotor_antiphase_correlation"] <= -0.95


def test_quality_report_is_schema_valid_and_exactly_replayed() -> None:
    validation = validate_report(REPORT)
    assert validation["passed"] is True and validation["record_count"] == 65


def test_flattened_action_curve_fails_closed() -> None:
    manifest = json.loads(SOURCE.read_text(encoding="utf-8"))
    attack = next(item for item in manifest["programs"][0]["clips"] if item["motion"] == "attack")
    for facing in attack["facings"]:
        for frame in facing["frames"]:
            frame["drivers"] = [0.0] * len(frame["drivers"])
    # The isolated metric itself is deliberately public enough to prove that
    # collapse is detected without publishing a forged bank.
    from forge.cellular_motion_quality import _clip_metrics
    metrics = _clip_metrics(attack)
    assert metrics["active_driver_count"] == 0
    assert metrics["group_excursion"]["appendage"] == 0.0
    assert metrics["maximum_pose_excursion"] == 0.0


def test_malformed_report_is_rejected(tmp_path: Path) -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8")); report["gates"]["actions_remain_articulated"] = False
    path = tmp_path / "tampered.json"; path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError):
        validate_report(path)
