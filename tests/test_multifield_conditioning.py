from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator
import pytest

from forge.multifield_conditioning import (
    CONDITIONING_AUDIT_FORMAT,
    conditioning_audit_source_hash,
    paired_classification_statistics,
)
from forge.multifield_conditioning.audit import _intervention_summary
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_paired_statistics_distinguish_proxy_ceiling_from_regression() -> None:
    report = paired_classification_statistics(
        [True, True, True, False, False, True],
        [True, True, False, True, False, True],
        [0, 1, 2, 0, 1, 2],
        [0, 1, 0, 2, 1, 2],
    )
    assert report["samples"] == 6
    assert report["both_correct"] == 3
    assert report["generated_only_correct"] == 1
    assert report["reference_only_correct"] == 1
    assert report["both_wrong"] == 1
    assert report["prediction_agreement"] == 4
    assert report["mcnemar_two_sided_exact_p"] == 1.0
    assert not report["generated_significantly_worse_at_0_01"]


def test_paired_statistics_detect_a_large_one_sided_regression() -> None:
    report = paired_classification_statistics(
        [False] * 12,
        [True] * 12,
        [0] * 12,
        [1] * 12,
    )
    assert report["reference_only_correct"] == 12
    assert report["generated_only_correct"] == 0
    assert report["mcnemar_two_sided_exact_p"] < 0.01
    assert report["generated_significantly_worse_at_0_01"]


def test_conditioning_audit_schema_and_source_hash_are_valid() -> None:
    schema_path = (
        PROJECT_ROOT / "shared" / "schema" / "multifield_conditioning_audit.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    assert CONDITIONING_AUDIT_FORMAT.endswith("-v3")
    assert len(conditioning_audit_source_hash()) == 64


def test_paired_statistics_reject_malformed_vectors() -> None:
    with pytest.raises(ValueError, match="non-empty and equal length"):
        paired_classification_statistics([], [], [], [])
    with pytest.raises(ValueError, match="non-empty and equal length"):
        paired_classification_statistics([True], [], [0], [0])


def test_same_noise_intervention_summary_separates_control_and_effect() -> None:
    baseline = tuple(torch.zeros((2, 4, 4), dtype=torch.long) for _ in range(3))
    control = tuple(values.clone() for values in baseline)
    unchanged = _intervention_summary(baseline, control)
    assert unchanged["exactly_unchanged"] == 2
    assert not unchanged["condition_effect_detected"]

    candidate = tuple(values.clone() for values in baseline)
    candidate[0][0, 1, 1] = 1
    changed = _intervention_summary(baseline, candidate)
    assert changed["exactly_changed"] == 1
    assert changed["condition_effect_detected"]
    assert changed["mean_categorical_hamming"] > 0.0
