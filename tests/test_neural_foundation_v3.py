from pathlib import Path

import pytest

from forge.neural_foundation_v3.release import build, validate


def test_foundation_build_and_validation(tmp_path: Path) -> None:
    result = build(tmp_path / "foundation")
    assert result["passed"] and result["components"] == 5
    assert result["recurrent_frames_per_second"] >= 30
    assert result["organism_physics_hz"] >= 12
    assert validate(tmp_path / "foundation") == result


def test_foundation_rejects_component_tamper(tmp_path: Path) -> None:
    output = tmp_path / "foundation"; build(output); path = output / "foundation_manifest.json"; data = path.read_bytes(); path.write_bytes(data.replace(b'"display_fps":30', b'"display_fps":29'))
    with pytest.raises(ValueError): validate(output)
