from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from forge.config import PROJECT_ROOT
from forge.creature_stage_neural_physiology.contract import source_sha256
from forge.creature_stage_neural_physiology.evaluation import (
    REPORT_NAME,
    _canonical,
    evaluate_checkpoint,
    evaluation_source_sha256,
    validate_evaluation,
)


SMOKE = PROJECT_ROOT / "outputs/creature_stage_neural_physiology/smoke_cpu_v1/smoke_checkpoint.pt"


def test_prediction_fed_physiology_diagnostic_is_exact_and_not_promotable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    output = tmp_path / "evaluation"
    result = evaluate_checkpoint(SMOKE, output, intervention_ids=(0, 4), rollout_frames=4)
    assert result["passed"]
    assert result["checkpoint_kind"] == "smoke"
    assert result["clips"] == 10
    assert not result["promotion_eligible"]
    assert result["gates"]["outside_cells_exact_zero"]
    assert validate_evaluation(output / REPORT_NAME, replay=True) == result


def test_rehashed_nested_metric_tamper_fails_derived_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    output = tmp_path / "evaluation"
    evaluate_checkpoint(SMOKE, output, intervention_ids=(0,), rollout_frames=2)
    path = output / REPORT_NAME
    payload = json.loads(path.read_bytes())
    payload["clips"][0]["metrics"]["health_mae"] += 0.125
    payload["semantic_sha256"] = hashlib.sha256(
        _canonical({key: value for key, value in payload.items() if key != "semantic_sha256"})
    ).hexdigest()
    path.write_bytes(_canonical(payload))
    with pytest.raises(ValueError, match="aggregate drifted"):
        validate_evaluation(path)


def test_evaluator_does_not_change_frozen_model_source() -> None:
    assert len(evaluation_source_sha256()) == 64
    assert source_sha256() == "e302f78f5989b21c52320cec5ed8ed6c5d8eed2cdf231f442018d37424a2bd68"
