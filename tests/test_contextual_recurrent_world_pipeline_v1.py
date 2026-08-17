import json
from pathlib import Path

import pytest

from forge.contextual_recurrent_world_pipeline_v1.contract import FORMAT, canonical
from forge.contextual_recurrent_world_pipeline_v1.release import validate


def test_contextual_release_validator_rejects_tamper(tmp_path: Path) -> None:
    payload = {"format": FORMAT, "status": "ready", "source_sha256": "bad", "gates": {"target_30fps": True}, "benchmark": {"frames_per_second": 60}, "parameters": 1, "manifest_sha256": "bad"}; path = tmp_path / "release.json"; path.write_bytes(canonical(payload))
    with pytest.raises(ValueError): validate(path)
