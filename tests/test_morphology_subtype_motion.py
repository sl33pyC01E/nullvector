from __future__ import annotations

import json
from pathlib import Path

import pytest

from forge.morphology_subtype_motion import DEFAULT_OUTPUT, _build, build_bank, validate_bank


@pytest.fixture(scope="module")
def compiled():
    return _build()


def test_all_twenty_subtypes_keep_full_motion_vocabulary_and_appendage_excursion(compiled) -> None:
    _, report = compiled
    assert report["status"] == "ready" and all(report["gates"].values())
    assert report["north_clip_count"] == 260 and report["directional_locomotion_clip_count"] == 160
    for identity in report["identities"]:
        assert identity["motion_count"] == 13
        assert all(identity["motion_gates"].values())
        assert identity["distinct_peak_pose_count"] >= 11


def test_every_subtype_has_directionally_distinct_locomotion(compiled) -> None:
    _, report = compiled
    assert all(len(item["directions"]) == 8 for item in report["directional_locomotion"])
    assert min(item["unique_directional_signatures"] for item in report["directional_locomotion"]) >= 6


def test_bank_is_exactly_replayable_and_tamper_evident(tmp_path: Path) -> None:
    destination = tmp_path / "motion"
    result = build_bank(destination)
    assert result["passed"] is True and result["subtype_count"] == 20
    assert validate_bank(destination / "morphology_subtype_motion.json") == result
    manifest = destination / "morphology_subtype_motion.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["identities"][0]["clips"][0]["layer_unique_frames"]["head"] += 1
    manifest.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    with pytest.raises(ValueError):
        validate_bank(manifest)


def test_published_bank_replays_when_present() -> None:
    manifest = DEFAULT_OUTPUT / "morphology_subtype_motion.json"
    if manifest.is_file():
        assert validate_bank(manifest)["passed"] is True
