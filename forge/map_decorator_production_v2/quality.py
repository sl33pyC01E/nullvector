from __future__ import annotations

import math
from typing import Any

from .contract import CALIBRATION_GATES, PRODUCTION_GATES, V2_CONTRACT_SHA256


def _density(metrics: dict[str, Any]) -> tuple[int, int, float | None]:
    target = sum(int(value) for value in metrics["target_count"][1:])
    prediction = sum(int(value) for value in metrics["prediction_count"][1:])
    return target, prediction, prediction / target if target else None


def evaluate_split_gate(
    metrics: dict[str, Any],
    *,
    stage: str,
) -> dict[str, object]:
    if stage == "calibration":
        contract = CALIBRATION_GATES
    elif stage == "production":
        contract = PRODUCTION_GATES
    else:
        raise ValueError("stage must be 'calibration' or 'production'.")
    failures: list[str] = []
    hard_checks = {
        "hard_legality": float(metrics.get("hard_legality", -1.0)) == float(contract["hard_legality"]),
        "immutable_semantic_changes": int(metrics.get("immutable_semantic_changes", -1))
        == int(contract["immutable_semantic_changes"]),
        "source_provenance_failures": int(metrics.get("source_provenance_failures", -1))
        == int(contract["source_provenance_failures"]),
    }
    for name, passed in hard_checks.items():
        if not passed:
            failures.append(name)
    head_checks: dict[str, object] = {}
    for head, thresholds in contract["heads"].items():  # type: ignore[union-attr]
        observed = metrics["heads"][head]
        checks = {
            "foreground_macro_iou": math.isfinite(float(observed["foreground_macro_iou"]))
            and float(observed["foreground_macro_iou"])
            >= float(thresholds["foreground_macro_iou_min"]),
            "foreground_f1": math.isfinite(float(observed["foreground_f1"]))
            and float(observed["foreground_f1"]) >= float(thresholds["foreground_f1_min"]),
            "rare_class_recall": math.isfinite(float(observed["rare_class_recall"]))
            and float(observed["rare_class_recall"])
            >= float(thresholds["rare_class_recall_min"]),
        }
        target, prediction, density_ratio = _density(observed) if head != "variant" else (0, 0, None)
        if "foreground_density_ratio" in thresholds:
            lower, upper = thresholds["foreground_density_ratio"]
            checks["foreground_density_ratio"] = (
                density_ratio is not None and float(lower) <= density_ratio <= float(upper)
            )
        if thresholds.get("every_target_foreground_class_predicted", False):
            checks["every_target_foreground_class_predicted"] = all(
                int(predicted) > 0
                for truth, predicted in zip(
                    observed["target_count"][1:], observed["prediction_count"][1:], strict=True
                )
                if int(truth) > 0
            )
        for check, passed in checks.items():
            if not passed:
                failures.append(f"{head}.{check}")
        head_checks[head] = {
            "thresholds": thresholds,
            "observed": {
                "foreground_macro_iou": float(observed["foreground_macro_iou"]),
                "foreground_f1": float(observed["foreground_f1"]),
                "rare_class_recall": float(observed["rare_class_recall"]),
                "target_foreground_cells": target,
                "predicted_foreground_cells": prediction,
                "foreground_density_ratio": density_ratio,
            },
            "checks": checks,
            "passed": all(checks.values()),
        }
    return {
        "v2_contract_sha256": V2_CONTRACT_SHA256,
        "stage": stage,
        "hard_checks": hard_checks,
        "head_checks": head_checks,
        "failures": failures,
        "passed": not failures,
    }


def evaluate_dual_split_gate(
    held_out: dict[str, Any],
    sentinel: dict[str, Any],
    *,
    stage: str,
) -> dict[str, object]:
    validation = evaluate_split_gate(held_out, stage=stage)
    test = evaluate_split_gate(sentinel, stage=stage)
    return {
        "v2_contract_sha256": V2_CONTRACT_SHA256,
        "stage": stage,
        "validation": validation,
        "test": test,
        "passed": bool(validation["passed"] and test["passed"]),
    }
