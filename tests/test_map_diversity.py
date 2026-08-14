from __future__ import annotations

import json
from pathlib import Path

import pytest

from forge.map_diversity import DEFAULT_OUTPUT, FEATURE_NAMES, audit_map_diversity, build_report, validate_report


def test_fixed_seed_matrix_proves_structural_not_only_byte_diversity() -> None:
    first = audit_map_diversity(); second = audit_map_diversity()
    assert first == second and first["status"] == "ready"
    assert len(FEATURE_NAMES) == 83 and first["aggregate"]["map_count"] == 48
    assert first["aggregate"]["unique_semantic_count"] == 48
    assert first["aggregate"]["unique_coarse_topology_count"] >= 44
    assert first["aggregate"]["leave_one_out_accuracy"] >= 0.85
    assert set(first["aggregate"]["per_theme"]) == {"arena", "rooms", "caves", "archipelago", "garden", "anomaly"}


def test_report_publication_is_canonical_and_exactly_replayable(tmp_path: Path) -> None:
    path = tmp_path / "map_diversity.json"; built = build_report(path)
    assert built["passed"] is True and validate_report(path) == built
    raw = path.read_bytes(); payload = json.loads(raw); payload["records"][0]["feature_values"][0] += 0.1
    path.write_text(json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError): validate_report(path)


def test_published_authority_replays_when_present() -> None:
    if DEFAULT_OUTPUT.is_file(): assert validate_report(DEFAULT_OUTPUT)["passed"] is True
