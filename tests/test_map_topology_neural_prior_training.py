from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from forge.map_topology_neural_prior.masking import MASK_MODES, mask_tokens
from forge.map_topology_neural_prior.model import build_prior, masked_token_loss
from forge.map_topology_neural_prior_training.checkpoint import load_checkpoint, save_checkpoint
from forge.map_topology_neural_prior_training.contract import (
    CHECKPOINT_FORMAT,
    FROZEN_LATENT_CORPUS_IDENTITY_SHA256,
    FROZEN_LATENT_CORPUS_MANIFEST_FILE_SHA256,
    PriorCalibrationConfig,
    canonical_json_bytes,
    source_manifest,
    training_source_sha256,
    validate_evaluation,
    validate_history,
)
from forge.map_topology_neural_prior_training.dataset import PriorTrainingDataset
from forge.map_topology_neural_prior_training.training import _quality
from forge.map_topology_neural_production.checkpoint import tensor_state_sha256


CORPUS = Path("outputs/map_decorator_corpus_v1")
LATENTS = Path("outputs/map_topology_neural_prior_corpus/v1")


@pytest.fixture(scope="module")
def dataset() -> PriorTrainingDataset:
    return PriorTrainingDataset(CORPUS, LATENTS)


def _metric(samples: int, accuracy: float = 0.2) -> dict:
    modes = {
        name: {"masked_cells": 6, "accuracy": accuracy + index * 0.01}
        for index, name in enumerate(MASK_MODES)
    }
    return {
        "sample_count": samples,
        "sample_registry_sha256": "1" * 64,
        "masked_cells": 24,
        "loss": 5.4,
        "accuracy": accuracy,
        "macro_mode_accuracy": sum(row["accuracy"] for row in modes.values()) / len(modes),
        "modes": modes,
        "vocabulary_size": 512,
    }


def _evaluation(samples: int = 6) -> dict:
    return {
        split: {mode: _metric(samples) for mode in ("baseline", "raw", "ema")}
        for split in ("validation", "test")
    }


def _history(config: PriorCalibrationConfig) -> list[dict]:
    return [
        {
            "step": step,
            "loss": 7.0 - step,
            "gradient_norm": 1.0,
            "batch_size": 4,
            "shape": [8, 8],
            "masked_cells": 80,
            "mask_fraction_mean": 0.4,
            "modes": list(MASK_MODES),
            "sample_registry_sha256": "2" * 64,
        }
        for step in range(1, config.steps + 1)
    ]


def test_training_contract_and_frozen_latent_split_are_exact(dataset: PriorTrainingDataset) -> None:
    assert len(training_source_sha256()) == 64
    assert set(source_manifest()) == {
        "forge/map_topology_neural_prior_training/__init__.py",
        "forge/map_topology_neural_prior_training/__main__.py",
        "forge/map_topology_neural_prior_training/checkpoint.py",
        "forge/map_topology_neural_prior_training/contract.py",
        "forge/map_topology_neural_prior_training/dataset.py",
        "forge/map_topology_neural_prior_training/metrics.py",
        "forge/map_topology_neural_prior_training/training.py",
    }
    assert {split: len(dataset.refs_by_split[split]) for split in ("train", "validation", "test")} == {
        "train": 2496, "validation": 576, "test": 24,
    }
    assert len(dataset.train_buckets) == 8
    config = PriorCalibrationConfig(steps=1, validation_samples=6, test_samples=6)
    refs = dataset.training_refs(0, torch.Generator().manual_seed(7), config)
    assert refs and {ref.split for ref in refs} == {"train"}
    assert not ({ref.full_map_identity_sha256 for ref in refs} & {ref.full_map_identity_sha256 for ref in dataset.refs_by_split["validation"]})


def test_streaming_batch_and_cpu_loss_are_finite(dataset: PriorTrainingDataset) -> None:
    config = PriorCalibrationConfig(steps=1, width=16, residual_depth=1, validation_samples=6, test_samples=6)
    refs = dataset.training_refs(3, torch.Generator().manual_seed(11), config)
    batch = dataset.collate(refs)
    assert batch["targets"].dtype == torch.long
    assert batch["valid_mask"].dtype == torch.bool
    assert batch["point_conditions"].dtype == torch.float32
    masked = mask_tokens(batch["targets"], batch["valid_mask"], generator=torch.Generator().manual_seed(13), config=config.model_config(), step=3)
    model = build_prior(config.model_config())
    logits = model({
        "tokens": masked["tokens"], "valid_mask": batch["valid_mask"],
        "point_conditions": batch["point_conditions"], "global_conditions": batch["global_conditions"],
        "theme_index": batch["theme_index"], "mask_fraction": masked["mask_fraction"],
    })
    loss = masked_token_loss(logits, batch["targets"], masked["mask"])
    loss.backward()
    assert bool(torch.isfinite(loss))
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_history_evaluation_and_quality_are_derived_fail_closed() -> None:
    config = PriorCalibrationConfig(steps=4, validation_samples=6, test_samples=6)
    history = _history(config)
    evaluation = _evaluation()
    validate_history(history, config)
    validate_evaluation(evaluation, config)
    validate_evaluation(json.loads(canonical_json_bytes(evaluation)), config)
    assert _quality(history, evaluation)["quality_milestone_reached"] is True
    broken = json.loads(json.dumps(evaluation))
    broken["validation"]["ema"]["modes"]["random"]["masked_cells"] += 1
    with pytest.raises(ValueError, match="totals"):
        validate_evaluation(broken, config)
    bad_history = json.loads(json.dumps(history))
    bad_history[0]["modes"] = ["random"]
    with pytest.raises(ValueError, match="mask-mode"):
        validate_history(bad_history, config)


def test_checkpoint_roundtrip_binds_source_corpus_metrics_and_sidecar(tmp_path: Path) -> None:
    config = PriorCalibrationConfig(steps=1, width=16, residual_depth=1, validation_samples=6, test_samples=6)
    model = build_prior(config.model_config())
    state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    state_sha = tensor_state_sha256(state)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "source_sha256": training_source_sha256(),
        "source_manifest": source_manifest(),
        "latent_corpus_manifest_file_sha256": FROZEN_LATENT_CORPUS_MANIFEST_FILE_SHA256,
        "latent_corpus_identity_sha256": FROZEN_LATENT_CORPUS_IDENTITY_SHA256,
        "config": config.to_dict(), "step": 1, "model_state": state, "ema_state": state,
        "optimizer_state": {}, "generator_state": torch.Generator().get_state(),
        "torch_cpu_rng_state": torch.get_rng_state(), "torch_cuda_rng_states": [],
        "history": _history(config), "evaluation": _evaluation(),
        "model_state_sha256": state_sha, "ema_state_sha256": state_sha,
    }
    path = tmp_path / "checkpoint.pt"
    sidecar = save_checkpoint(path, payload)
    loaded = load_checkpoint(path)
    assert loaded["model_state_sha256"] == state_sha
    assert sidecar["sha256"] and sidecar["sidecar_sha256"]
    sidecar_path = path.with_suffix(".pt.json")
    original = sidecar_path.read_bytes()
    try:
        tampered = json.loads(original)
        tampered["step"] = 99
        sidecar_path.write_text(json.dumps(tampered), encoding="utf-8")
        with pytest.raises(ValueError, match="self-hash"):
            load_checkpoint(path)
    finally:
        sidecar_path.write_bytes(original)
