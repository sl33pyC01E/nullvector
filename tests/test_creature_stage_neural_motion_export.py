from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from forge.config import PROJECT_ROOT
from forge.creature_stage_neural_motion.contract import source_sha256
from forge.creature_stage_neural_motion.export import (
    MANIFEST_NAME,
    _canonical,
    export_checkpoint,
    export_source_sha256,
    validate_export,
)


SMOKE_CHECKPOINT = PROJECT_ROOT / "outputs/creature_stage_neural_motion/smoke_cpu_v1_final/smoke_checkpoint.pt"


@pytest.fixture(scope="module")
def portable_export(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("cell-motion-onnx") / "export"
    result = export_checkpoint(SMOKE_CHECKPOINT, output)
    assert result["passed"]
    return output


def test_portable_export_has_dynamic_batch_and_heldout_parity(portable_export: Path) -> None:
    result = validate_export(portable_export)
    assert result["passed"]
    assert result["checkpoint_kind"] == "smoke"
    assert result["examples"] == 80
    assert result["max_abs"] <= 5e-5
    assert result["mean_abs"] <= 5e-6
    assert all(result["gates"].values())


def test_rehashed_parity_tamper_fails_numerical_replay(portable_export: Path) -> None:
    path = portable_export / MANIFEST_NAME
    original = path.read_bytes()
    payload = json.loads(original)
    payload["parity"]["cases"][0]["max_abs"] += 0.000001
    payload["semantic_sha256"] = hashlib.sha256(
        _canonical({key: value for key, value in payload.items() if key != "semantic_sha256"})
    ).hexdigest()
    path.write_bytes(_canonical(payload))
    try:
        with pytest.raises(ValueError, match="numerical replay drifted"):
            validate_export(portable_export)
    finally:
        path.write_bytes(original)


def test_onnx_bytes_are_checkpoint_bound(portable_export: Path) -> None:
    path = portable_export / "cellular_motion.onnx"
    original = path.read_bytes()
    damaged = bytearray(original)
    damaged[-1] ^= 1
    path.write_bytes(damaged)
    try:
        with pytest.raises(ValueError, match="artifact bytes drifted"):
            validate_export(portable_export)
    finally:
        path.write_bytes(original)


def test_export_is_additive_to_frozen_training_source() -> None:
    assert len(export_source_sha256()) == 64
    assert source_sha256() == "2300cacade824488a69d1f191519e5809222f1de14ecd8d92f64f3ea1f3b5ec5"
