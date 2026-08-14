from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from forge.map_topology_neural.codec import build_codec
from forge.map_topology_neural.contract import CONTRACT_SHA256
from forge.map_topology_neural.corpus import FROZEN_CORPUS_MANIFEST_FILE_SHA256, FROZEN_CORPUS_SHA256
from forge.map_topology_neural_production.checkpoint import (
    load_checkpoint,
    save_checkpoint,
    tensor_state_sha256,
)
from forge.map_topology_neural_production.contract import (
    CHECKPOINT_FORMAT,
    QUALITY_GATES,
    TopologyCodecCalibrationConfig,
    production_source_manifest,
    production_source_sha256,
)
from forge.map_topology_neural_production.dataset import TopologyProductionDataset
from forge.map_topology_neural_production.metrics import balanced_reconstruction_loss
from forge.map_topology_neural_production.training import (
    CHECKPOINT_REPORT_KEYS,
    REPORT_KEYS,
    RUNTIME_KEYS,
    SAFETY_GATE_KEYS,
    _validate_report_contract,
)


CORPUS = Path("outputs/map_decorator_corpus_v1")


@pytest.fixture(scope="module")
def dataset() -> TopologyProductionDataset:
    return TopologyProductionDataset(CORPUS)


def test_topology_production_contract_is_bounded_and_source_complete() -> None:
    config = TopologyCodecCalibrationConfig()
    assert config.steps == 100
    assert config.validation_samples == 48
    assert config.test_samples == 24
    assert config.codec_config().codebook_size == 512
    assert QUALITY_GATES["minimum_walkability_iou"] == 0.65
    with pytest.raises(ValueError, match=r"\[1,500\]"):
        TopologyCodecCalibrationConfig(steps=501)
    manifest = production_source_manifest()
    assert set(manifest) >= {
        "forge/map_topology_neural/codec.py",
        "forge/map_topology_neural_production/training.py",
        "forge/map_topology_neural_production/checkpoint.py",
    }
    assert len(production_source_sha256()) == 64


def test_topology_production_dataset_has_exact_split_shape_and_eval_census(
    dataset: TopologyProductionDataset,
) -> None:
    assert {name: len(refs) for name, refs in dataset.refs_by_split.items()} == {
        "train": 2496,
        "validation": 576,
        "test": 24,
    }
    assert len(dataset.train_buckets) == 8
    assert len(dataset.evaluation_refs("validation", 48)) == 48
    assert len(dataset.evaluation_refs("validation", 6)) == 6
    test_refs = dataset.evaluation_refs("test", 24)
    assert len(test_refs) == 24
    assert len(dataset.evaluation_refs("test", 6)) == 6
    assert {ref.theme for ref in test_refs} == {"arena", "rooms", "caves", "archipelago", "garden", "anomaly"}


def test_topology_production_batch_and_balanced_loss_are_finite(
    dataset: TopologyProductionDataset,
) -> None:
    config = TopologyCodecCalibrationConfig(
        steps=1,
        codec_width=8,
        latent_dim=8,
        codebook_size=16,
        field_embedding_dim=2,
        residual_depth=0,
    )
    generator = torch.Generator().manual_seed(config.seed)
    refs = dataset.training_refs(0, generator, config)
    batch = dataset.collate(refs[:1], torch.device("cpu"))
    model = build_codec(config.codec_config(), init_seed=config.seed)
    output = model(batch, update_ema=True)
    losses = balanced_reconstruction_loss(output, batch)
    losses["total"].backward()
    assert all(bool(torch.isfinite(value)) for value in losses.values())
    assert any(parameter.grad is not None for parameter in model.parameters())


def _checkpoint_payload(config: TopologyCodecCalibrationConfig, registry: str):
    model = build_codec(config.codec_config(), init_seed=config.seed)
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    return {
        "format": CHECKPOINT_FORMAT,
        "source_sha256": production_source_sha256(),
        "source_manifest": production_source_manifest(),
        "tensor_contract_sha256": CONTRACT_SHA256,
        "corpus_sha256": FROZEN_CORPUS_SHA256,
        "corpus_manifest_file_sha256": FROZEN_CORPUS_MANIFEST_FILE_SHA256,
        "dataset_registry_sha256": registry,
        "config": config.to_dict(),
        "step": config.steps,
        "model_state": state,
        "ema_state": {name: value.clone() for name, value in state.items()},
        "optimizer_state": {},
        "training_generator_state": torch.Generator().manual_seed(7).get_state(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_states": [],
        "history": [{"step": 1, "loss": {"total": 1.0}}],
        "evaluation": {},
        "model_state_sha256": tensor_state_sha256(state),
        "ema_state_sha256": tensor_state_sha256(state),
    }


def test_topology_production_checkpoint_round_trip_and_sidecar_tamper(
    dataset: TopologyProductionDataset,
    tmp_path: Path,
) -> None:
    config = TopologyCodecCalibrationConfig(
        steps=1,
        codec_width=8,
        latent_dim=8,
        codebook_size=16,
        field_embedding_dim=2,
        residual_depth=0,
    )
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, _checkpoint_payload(config, dataset.registry_sha256))
    assert load_checkpoint(path)["dataset_registry_sha256"] == dataset.registry_sha256
    sidecar_path = path.with_suffix(".pt.json")
    original = sidecar_path.read_bytes()
    try:
        sidecar = json.loads(original)
        sidecar["step"] = 2
        sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
        with pytest.raises(ValueError, match="self-hash"):
            load_checkpoint(path)
    finally:
        sidecar_path.write_bytes(original)


def test_topology_production_report_contract_rejects_rehashed_claim_drift() -> None:
    safety = {name: True for name in SAFETY_GATE_KEYS}
    quality = {"quality_milestone_reached": True}
    report = {name: None for name in REPORT_KEYS}
    report.update({
        "format": "nullvector-neural-map-topology-codec-calibration/1.0.0",
        "status": "passed",
        "tensor_contract_sha256": CONTRACT_SHA256,
        "quality": quality,
        "safety_gates": safety,
        "claim_boundary": {
            "representation_calibration_only": True,
            "quality_milestone_reached": True,
            "generative_prior_trained": False,
            "compiled_map_bank_published": False,
            "godot_integration": False,
        },
        "runtime": {
            "device": "test",
            "compute_capability": [8, 9],
            "precision": "bf16-autocast-float32-loss",
            "training_seconds": 1.0,
            "evaluation_seconds": 1.0,
            "elapsed_seconds": 2.1,
            "peak_allocated_bytes": 1,
            "peak_reserved_bytes": 2,
        },
        "checkpoint": {name: None for name in CHECKPOINT_REPORT_KEYS},
    })
    _validate_report_contract(report)
    report["claim_boundary"]["generative_prior_trained"] = True
    with pytest.raises(ValueError, match="claim boundary"):
        _validate_report_contract(report)
    report["claim_boundary"]["generative_prior_trained"] = False
    report["safety_gates"]["invented_gate"] = True
    with pytest.raises(ValueError, match="safety gate census"):
        _validate_report_contract(report)
