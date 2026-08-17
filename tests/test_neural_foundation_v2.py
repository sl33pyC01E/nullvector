from __future__ import annotations

from forge.neural_foundation_v2 import validate


def test_build_three_manifest_replays_when_available():
    result = validate()
    assert result["passed"]
    assert result["build"] == 3
    assert result["components"] == 4
