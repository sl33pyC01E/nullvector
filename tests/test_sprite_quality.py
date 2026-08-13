from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from forge.sprite_quality import (
    assert_exact_sprite_quality_replay,
    assert_valid_sprite_quality_report,
    audit_source_hash,
    build_sprite_quality_report,
    compile_sprite_quality_audit,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def report() -> dict:
    return build_sprite_quality_report()


def test_production_sprite_audit_has_exact_coverage(report: dict) -> None:
    assert report["coverage"]["static_sample_count"] == 80
    assert [item["count"] for item in report["coverage"]["family_counts"]] == [16] * 5
    assert [item["count"] for item in report["coverage"]["subtype_counts"]] == [4] * 20
    assert [item["count"] for item in report["coverage"]["role_counts"]] == [10] * 8
    assert all(item["count"] == 2 for item in report["coverage"]["family_role_cells"])
    assert report["coverage"]["motion_identity_count"] == 5
    assert report["coverage"]["all_80_motion_claimed"] is False


def test_production_categorical_bank_is_diverse_and_uses_every_vocab(report: dict) -> None:
    quality = report["categorical_quality"]
    assert quality["unique_categorical_field_count"] == 80
    assert quality["unique_visible_silhouette_count"] >= 75
    assert quality["pairwise_aligned_categorical_hamming"]["median"] > 0.20
    assert all(item["observed_class_count"] == item["class_count"] for item in quality["vocabularies"])


def test_production_motion_has_no_static_or_collapsed_clips(report: dict) -> None:
    motion = report["motion_quality"]
    assert motion["clip_count"] == 520
    assert motion["stored_frame_count"] == 4720
    assert motion["collapsed_base_clip_count"] == 0
    assert motion["collapsed_composite_clip_count"] == 0
    assert motion["minimum_unique_base_frames"] >= 4
    assert motion["minimum_unique_composite_frames"] >= 5
    assert motion["all_loop_endpoints_exact"]


def test_motion_diagnostics_distinguish_quiet_and_forceful_actions(report: dict) -> None:
    by_motion: dict[str, list[float]] = {}
    for row in report["motion_quality"]["by_family_motion"]:
        by_motion.setdefault(row["motion"], []).append(row["composite_change"]["mean"])
    assert sum(by_motion["sleep"]) / 5 < sum(by_motion["locomote"]) / 5
    assert sum(by_motion["idle_breathe"]) / 5 < sum(by_motion["attack"]) / 5
    assert sum(by_motion["idle_breathe"]) / 5 < sum(by_motion["death"]) / 5


def test_report_hash_and_source_hash_fail_closed(report: dict) -> None:
    assert len(audit_source_hash()) == 64
    assert_valid_sprite_quality_report(report)
    tampered = deepcopy(report)
    tampered["coverage"]["motion_identity_scope"] = "all eighty"
    with pytest.raises(ValueError, match="hash"):
        assert_valid_sprite_quality_report(tampered)


def test_immutable_compile_and_exact_replay(tmp_path: Path) -> None:
    destination = tmp_path / "audit"
    report = compile_sprite_quality_audit(destination)
    assert report["status"] == "passed"
    replay = assert_exact_sprite_quality_replay(destination / "sprite_quality_report.json")
    assert replay["exact_report"]
    assert replay["exact_heatmap"]
    with pytest.raises(FileExistsError):
        compile_sprite_quality_audit(destination)
