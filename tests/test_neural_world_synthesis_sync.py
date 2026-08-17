import json
from pathlib import Path

import pytest

from forge.neural_world_synthesis_sync import DEFAULT_SOURCE, FORMAT, project_runtime, validate_runtime


def test_runtime_projection_is_small_closed_and_cadence_bound() -> None:
    if not DEFAULT_SOURCE.is_dir():
        pytest.skip("Current neural world synthesis output is not present.")
    first = project_runtime(DEFAULT_SOURCE)
    second = project_runtime(DEFAULT_SOURCE)
    assert first == second
    assert set(first) == {"catalog.json", "neural_world_atlas.png"}
    catalog = json.loads(first["catalog.json"])
    assert catalog["format"] == FORMAT
    assert catalog["theme_count"] == 6
    assert catalog["layer_count"] == 8
    assert catalog["atlas_frame_count"] == 90
    assert catalog["runtime"]["display_target_fps"] == 30
    assert catalog["runtime"]["embodied_motion_hz"] == 30
    assert catalog["runtime"]["causal_world_hz"] == 15
    assert catalog["world_synthesis_in_frame_loop"] is False
    assert catalog["python_runtime_required"] is False
    assert catalog["cuda_runtime_required"] is False


def test_runtime_validation_detects_atlas_tamper(tmp_path: Path) -> None:
    if not DEFAULT_SOURCE.is_dir():
        pytest.skip("Current neural world synthesis output is not present.")
    files = project_runtime(DEFAULT_SOURCE)
    for relative, payload in files.items():
        (tmp_path / relative).write_bytes(payload)
    assert validate_runtime(tmp_path, DEFAULT_SOURCE)["passed"] is True
    atlas = tmp_path / "neural_world_atlas.png"
    atlas.write_bytes(atlas.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="artifact replay"):
        validate_runtime(tmp_path, DEFAULT_SOURCE)
