from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from forge.map_topology_neural_prior_v2.masking import MASK_MODES_V2
from forge.map_topology_neural_prior_v2.model import build_prior_v2
from forge.map_topology_neural_prior_v2_training.checkpoint import load_checkpoint
from forge.map_topology_neural_prior_v2_training.contract import (
    FROZEN_AUTHORITY, PriorV2CalibrationConfig, source_manifest,
    training_v2_source_sha256,
)
from forge.map_topology_neural_prior_v2_training.metrics import evaluate_free_generation, evaluate_masked
from forge.map_topology_neural_prior_v2_training.training import run_segment, validate_segment
from forge.map_topology_neural_prior_training.dataset import PriorTrainingDataset


CORPUS = Path("outputs/map_decorator_corpus_v1")
LATENTS = Path("outputs/map_topology_neural_prior_corpus/v1")


@pytest.fixture(scope="module")
def dataset() -> PriorTrainingDataset:
    return PriorTrainingDataset(CORPUS, LATENTS)


def tiny_config(*, segment_steps: int = 1) -> PriorV2CalibrationConfig:
    return PriorV2CalibrationConfig(
        total_steps=2, steps_per_segment=segment_steps, width=16,
        levels=2, blocks_per_level=1, sampling_steps=2,
        validation_samples=6, test_samples=6,
    )


def test_v2_training_contract_binds_architecture_and_frozen_corpus() -> None:
    assert len(training_v2_source_sha256()) == 64
    assert set(source_manifest()) == {
        "forge/map_topology_neural_prior_v2_training/__init__.py",
        "forge/map_topology_neural_prior_v2_training/__main__.py",
        "forge/map_topology_neural_prior_v2_training/checkpoint.py",
        "forge/map_topology_neural_prior_v2_training/contract.py",
        "forge/map_topology_neural_prior_v2_training/metrics.py",
        "forge/map_topology_neural_prior_v2_training/training.py",
    }
    assert set(FROZEN_AUTHORITY) == {"latent_corpus_manifest_file_sha256", "latent_corpus_identity_sha256"}
    with pytest.raises(ValueError, match="step bounds"):
        PriorV2CalibrationConfig(total_steps=2, steps_per_segment=3)


def test_full_mask_is_reported_separately_from_partial_modes(dataset: PriorTrainingDataset) -> None:
    config = tiny_config(); refs = dataset.evaluation_refs("validation", 6); model = build_prior_v2(config.model_config()).eval()
    metric = evaluate_masked(model, dataset, refs, device=torch.device("cpu"), config=config)
    assert set(metric["modes"]) == set(MASK_MODES_V2)
    assert metric["full_mask_accuracy"] == metric["modes"]["full"]["accuracy"]
    assert metric["full_mask_loss"] == metric["modes"]["full"]["loss"]
    assert all(metric["modes"][mode]["masked_cells"] > 0 for mode in MASK_MODES_V2)
    generated = evaluate_free_generation(model, dataset, refs, device=torch.device("cpu"), config=config)
    assert generated["sample_count"] == 6 and generated["all_tokens_revealed"] is True
    assert generated["unique_samples"] >= 2


def test_segment_resume_is_exactly_equivalent_to_uninterrupted_cpu_training(tmp_path: Path) -> None:
    segmented = tiny_config(segment_steps=1); direct = tiny_config(segment_steps=2)
    first = run_segment(CORPUS, LATENTS, tmp_path / "segment-1", config=segmented, device_name="cpu")
    second = run_segment(CORPUS, LATENTS, tmp_path / "segment-2", config=segmented, resume=tmp_path / "segment-1/checkpoint.pt", device_name="cpu")
    uninterrupted = run_segment(CORPUS, LATENTS, tmp_path / "direct", config=direct, device_name="cpu")
    assert first["segment"] == {"start_step": 0, "end_step": 1, "updates": 1, "predecessor": None}
    assert second["segment"]["predecessor"]["step"] == 1
    assert second["history"] == uninterrupted["history"]
    assert second["model"] == uninterrupted["model"]
    assert second["free_generation"] == uninterrupted["free_generation"]
    assert second["evaluation"] == uninterrupted["evaluation"]
    assert validate_segment(tmp_path / "segment-2") == second


def test_checkpoint_and_report_tamper_fail_closed(tmp_path: Path) -> None:
    output = tmp_path / "segment"
    run_segment(CORPUS, LATENTS, output, config=PriorV2CalibrationConfig(total_steps=1, steps_per_segment=1, width=16, levels=2, blocks_per_level=1, sampling_steps=2, validation_samples=6, test_samples=6), device_name="cpu")
    checkpoint = output / "checkpoint.pt"; sidecar = checkpoint.with_suffix(".pt.json"); original_sidecar = sidecar.read_bytes()
    try:
        payload = json.loads(original_sidecar); payload["step"] = 99; sidecar.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="self-hash"):
            load_checkpoint(checkpoint)
    finally: sidecar.write_bytes(original_sidecar)
    report_path = output / "segment_report.json"; original_report = report_path.read_bytes()
    try:
        report = json.loads(original_report); report["calibration"]["production_promotion_allowed"] = True; report_path.write_text(json.dumps(report), encoding="utf-8")
        with pytest.raises(ValueError, match="canonical JSON|self-hash"):
            validate_segment(output)
    finally: report_path.write_bytes(original_report)


def test_resume_rejects_config_drift(tmp_path: Path) -> None:
    config = tiny_config(); run_segment(CORPUS, LATENTS, tmp_path / "first", config=config, device_name="cpu")
    drifted = PriorV2CalibrationConfig(total_steps=2, steps_per_segment=1, width=24, levels=2, blocks_per_level=1, sampling_steps=2, validation_samples=6, test_samples=6)
    with pytest.raises(ValueError, match="incompatible"):
        run_segment(CORPUS, LATENTS, tmp_path / "rejected", config=drifted, resume=tmp_path / "first/checkpoint.pt", device_name="cpu")
