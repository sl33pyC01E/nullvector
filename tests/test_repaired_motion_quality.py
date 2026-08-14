from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from forge.multifield_style_motion.hashing import canonical_json_bytes, sha256_bytes
from forge.repaired_motion_quality import (
    DEFAULT_RUNTIME,
    REPORT_NAME,
    build_quality_audit,
    validate_quality_audit,
)


@pytest.fixture(scope="module")
def quality_bank(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, object]]:
    output = tmp_path_factory.mktemp("repaired-motion-quality") / "bank"
    result = build_quality_audit(output, DEFAULT_RUNTIME, visually_inspected=True)
    return output, result


def test_all_80_motion_quality_bank_is_dynamic_and_exact(quality_bank: tuple[Path, dict[str, object]]) -> None:
    output, result = quality_bank
    assert result["passed"]
    assert result["clip_count"] == 8_320
    assert result["frame_count"] == 75_520
    report = json.loads((output / REPORT_NAME).read_text())
    assert report["counts"]["unique_base_atlas_count"] == 80
    assert report["summary"] == {
        "articulation_failure_count": 0,
        "blank_clip_count": 0,
        "clip_metrics_sha256": result["clip_metrics_sha256"],
        "cross_identity_duplicate_alpha_sequence_group_count": 104,
        "cross_identity_duplicate_rgba_sequence_group_count": 0,
        "maximum_articulation_peak_ppm": 1_235_294,
        "maximum_occupancy_spike_ppm": 1_307_918,
        "minimum_articulation_peak_ppm": 36_680,
        "minimum_unique_frame_count": 4,
        "occupancy_spike_failure_count": 0,
        "static_clip_count": 0,
    }
    assert report["motions"]["attack"]["minimum_articulation_peak_ppm"] >= 320_000
    assert report["motions"]["locomote"]["minimum_articulation_peak_ppm"] >= 85_000
    assert report["motions"]["idle_breathe"]["minimum_articulation_peak_ppm"] >= 30_000
    assert all(report["gates"].values())
    assert validate_quality_audit(output, runtime=DEFAULT_RUNTIME) == result


def test_fully_rehashed_metric_tamper_fails_semantic_replay(quality_bank: tuple[Path, dict[str, object]], tmp_path: Path) -> None:
    source, _ = quality_bank
    tampered = tmp_path / "tampered"
    shutil.copytree(source, tampered)
    report_path = tampered / REPORT_NAME
    report = json.loads(report_path.read_text())
    report["summary"]["minimum_articulation_peak_ppm"] += 1
    unsigned = dict(report)
    unsigned.pop("report_sha256")
    report["report_sha256"] = sha256_bytes(canonical_json_bytes(unsigned))
    report_path.write_bytes(canonical_json_bytes(report))
    with pytest.raises(ValueError, match="exact semantic replay"):
        validate_quality_audit(tampered, runtime=DEFAULT_RUNTIME)
