from __future__ import annotations

import os

import pytest

from forge.map_decorator.hashing import json_sha256
from forge.map_decorator_production_v4_calibration.contract import (
    CalibrationConfig,
    V4_CALIBRATION_CONTRACT_SHA256,
    calibration_contract_manifest,
)
from forge.map_decorator_production_v4_calibration.evaluation import compare_to_baseline
from forge.map_decorator_production_v4_calibration.runner import (
    _config_from_dict,
    _configure_cuda,
    calibration_source_manifest,
    calibration_source_sha256,
)
from forge.map_decorator_production_v4_training.checkpoint import training_source_sha256


def _metrics(iou: float, f1: float, recall: float, *, legal: float = 1.0) -> dict[str, object]:
    heads = {
        name: {
            "foreground_macro_iou": iou,
            "foreground_f1": f1,
            "rare_class_recall": recall,
        }
        for name in ("decal", "prop")
    }
    return {
        "heads": heads,
        "hard_legality": legal,
        "immutable_semantic_changes": 0,
        "source_provenance_failures": 0,
        "full_split": True,
    }


def test_v4_calibration_contract_is_baseline_bound_and_nonregressing() -> None:
    manifest = calibration_contract_manifest()
    assert V4_CALIBRATION_CONTRACT_SHA256 == json_sha256(manifest)
    assert manifest["baseline"]["same_initial_model"] is True
    assert manifest["acceptance"]["raw_and_ema_each_nonregressing"] is True
    assert manifest["acceptance"]["at_least_one_strict_object_metric_improvement"] is True
    with pytest.raises(ValueError, match="steps"):
        CalibrationConfig(steps=0)


def test_v4_baseline_comparison_requires_every_metric_and_real_improvement() -> None:
    baseline = {split: _metrics(0.9, 0.94, 0.91) for split in ("validation", "test")}
    improved = {split: _metrics(0.91, 0.95, 0.92) for split in ("validation", "test")}
    result = compare_to_baseline(baseline, improved)
    assert result["passed"] is True and result["strict_improvement_count"] == 12
    unchanged = compare_to_baseline(baseline, baseline)
    assert unchanged["every_object_metric_nonregressing"] is True
    assert unchanged["passed"] is False
    regressed = {split: _metrics(0.899, 0.95, 0.92) for split in ("validation", "test")}
    result = compare_to_baseline(baseline, regressed)
    assert result["every_object_metric_nonregressing"] is False and result["passed"] is False
    unsafe = {split: _metrics(0.91, 0.95, 0.92, legal=0.0) for split in ("validation", "test")}
    assert compare_to_baseline(baseline, unsafe)["passed"] is False


def test_v4_calibration_config_round_trips_canonically() -> None:
    config = CalibrationConfig(steps=7, validation_batch_size=3, test_batch_size=2)
    assert _config_from_dict(config.to_dict()) == config
    malformed = config.to_dict()
    malformed["precision"] = "fp32"
    with pytest.raises(ValueError, match="precision"):
        _config_from_dict(malformed)


def test_v4_calibration_source_binds_immutable_training_source() -> None:
    manifest = calibration_source_manifest()
    assert manifest["training_source_sha256"] == training_source_sha256()
    assert calibration_source_sha256() == json_sha256(manifest)
    assert any(path.endswith("/runner.py") for path in manifest["calibration_files"])


def test_v4_cuda_setup_fails_before_initialization_without_workspace_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    with pytest.raises(RuntimeError, match="CUBLAS_WORKSPACE_CONFIG"):
        _configure_cuda(CalibrationConfig().training.seed)
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") is None
