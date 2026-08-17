from __future__ import annotations

import json

from forge.recurrent_world_rollout_v1.contract import DEFAULT_OUTPUT, canonical, source_sha256


def test_long_horizon_report_is_current_and_passes():
    raw = (DEFAULT_OUTPUT / "report.json").read_bytes();report = json.loads(raw)
    assert raw == canonical(report)
    assert report["source_sha256"] == source_sha256()
    assert report["status"] == "long_horizon_ready"
    assert report["gates"]["all_passed"]
