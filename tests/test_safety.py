from __future__ import annotations

import json

import pytest

from forge.safety import disk_status, require_disk_floor, write_json_atomic


def test_disk_floor_reports_and_rejects_impossible_reserve(tmp_path) -> None:
    status = require_disk_floor(tmp_path, floor_gb=0.0, planned_bytes=1024)
    assert status.safe
    assert status.free_gb > 0
    impossible = disk_status(tmp_path, floor_gb=status.total_gb + 1.0)
    assert not impossible.safe
    with pytest.raises(RuntimeError, match="Disk safety floor reached"):
        require_disk_floor(tmp_path, floor_gb=status.total_gb + 1.0)


def test_atomic_json_replaces_complete_payload(tmp_path) -> None:
    destination = tmp_path / "nested" / "artifact.json"
    write_json_atomic(destination, {"version": 1, "values": [1, 2, 3]})
    write_json_atomic(destination, {"version": 2, "valid": True})
    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "version": 2,
        "valid": True,
    }
    assert not destination.with_name(destination.name + ".tmp").exists()
