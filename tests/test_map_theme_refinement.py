from __future__ import annotations

import json
from pathlib import Path

import pytest

from forge.map_theme_refinement import DEFAULT_OUTPUT, _build, archipelago_policy_passes, build_bank, generate_refined_archipelago, validate_bank
from forge.maps import MapConfig, assert_valid
from forge.maps.generator import splitmix64


def test_bounded_refinement_is_deterministic_hard_valid_and_policy_compliant() -> None:
    cfg = MapConfig(width=48, height=48, objective_count=3, spawn_count=8, min_start_exit_distance=24)
    for index in range(128):
        seed = splitmix64(0xA11CE1A6 ^ index)
        first = generate_refined_archipelago(seed, cfg); second = generate_refined_archipelago(seed, cfg)
        assert (first.selected_seed, first.attempt, first.metrics) == (second.selected_seed, second.attempt, second.metrics)
        assert first.data.map_id == second.data.map_id
        assert archipelago_policy_passes(first.metrics)
        assert assert_valid(first.data)["failures"] == []


def test_refined_six_theme_matrix_has_exact_theme_separation() -> None:
    _, report = _build()
    assert report["status"] == "ready" and all(report["gates"].values())
    assert report["aggregate"]["leave_one_out_accuracy"] == 1.0
    assert all(value["recall"] == 1.0 for value in report["aggregate"]["per_theme"].values())


def test_bank_is_canonical_exactly_replayable_and_tamper_evident(tmp_path: Path) -> None:
    destination = tmp_path / "refined"
    result = build_bank(destination)
    assert result["passed"] is True and result["archipelago_recall"] == 1.0
    assert validate_bank(destination / "map_theme_refinement.json") == result
    manifest = destination / "map_theme_refinement.json"
    payload = json.loads(manifest.read_text(encoding="utf-8")); payload["records"][0]["attempt"] += 1
    manifest.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    with pytest.raises(ValueError): validate_bank(manifest)


def test_published_bank_replays_when_present() -> None:
    manifest = DEFAULT_OUTPUT / "map_theme_refinement.json"
    if manifest.is_file(): assert validate_bank(manifest)["passed"] is True
