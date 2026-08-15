from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from forge.config import PROJECT_ROOT
from forge.creature_stage_neural_motion.contract import source_sha256
from forge.creature_stage_neural_motion.evaluation import (
    REPORT_NAME,
    _canonical,
    _cross_backend_difference,
    evaluate_checkpoint,
    evaluation_source_sha256,
    validate_evaluation,
)
from forge.creature_stage_neural_motion.training import validate_cpu_smoke


SMOKE_ROOT = PROJECT_ROOT / "outputs/creature_stage_neural_motion/smoke_cpu_v1_final"


def test_prediction_fed_diagnostic_rollout_is_exact_and_not_promotable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    output = tmp_path / "evaluation"
    result = evaluate_checkpoint(
        SMOKE_ROOT / "smoke_checkpoint.pt",
        output,
        motion_ids=(0, 9),
        rollout_frames=4,
    )
    assert result["passed"]
    assert result["checkpoint_kind"] == "smoke"
    assert result["clips"] == 10
    assert not result["promotion_eligible"]
    assert not result["gates"]["full_validation_matrix"]
    assert not result["gates"]["final_production_checkpoint"]
    assert result["gates"]["outside_cells_exact_zero"]
    assert validate_evaluation(output / REPORT_NAME, replay=True) == result


def test_rehashed_nested_metric_tamper_fails_derived_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    output = tmp_path / "evaluation"
    evaluate_checkpoint(
        SMOKE_ROOT / "smoke_checkpoint.pt",
        output,
        motion_ids=(0,),
        rollout_frames=2,
    )
    path = output / REPORT_NAME
    payload = json.loads(path.read_bytes())
    payload["clips"][0]["metrics"]["position_mae_px"] += 0.125
    payload["semantic_sha256"] = hashlib.sha256(
        _canonical({key: value for key, value in payload.items() if key != "semantic_sha256"})
    ).hexdigest()
    path.write_bytes(_canonical(payload))
    with pytest.raises(ValueError, match="aggregate drifted"):
        validate_evaluation(path)


def test_cross_backend_replay_is_structurally_exact_and_numerically_bounded() -> None:
    expected = {
        "source": "a" * 64,
        "gates": {"finite": True},
        "rows": [{"metric": 1.0, "count": 5}],
    }
    within = {
        "source": "a" * 64,
        "gates": {"finite": True},
        "rows": [{"metric": 1.000019, "count": 5}],
    }
    assert _cross_backend_difference(within, expected) is None

    changed_gate = json.loads(json.dumps(within))
    changed_gate["gates"]["finite"] = False
    assert "$.gates.finite" in _cross_backend_difference(changed_gate, expected)

    changed_count = json.loads(json.dumps(within))
    changed_count["rows"][0]["count"] = 6
    assert "$.rows[0].count" in _cross_backend_difference(changed_count, expected)

    outside = json.loads(json.dumps(within))
    outside["rows"][0]["metric"] = 1.001
    assert "replay tolerance" in _cross_backend_difference(outside, expected)


def test_sealed_test_split_requires_explicit_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    with pytest.raises(PermissionError, match="explicit release"):
        evaluate_checkpoint(
            SMOKE_ROOT / "smoke_checkpoint.pt",
            tmp_path / "evaluation",
            split="test",
            motion_ids=(0,),
            rollout_frames=2,
        )


def test_evaluator_is_additive_to_frozen_model_source() -> None:
    assert len(evaluation_source_sha256()) == 64
    assert source_sha256() == "2300cacade824488a69d1f191519e5809222f1de14ecd8d92f64f3ea1f3b5ec5"
    assert validate_cpu_smoke(SMOKE_ROOT)["passed"]
