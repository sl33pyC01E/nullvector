from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from forge.morphology import allowed_training_field_tuples
from forge.sprite_latent import SemanticSpriteFSQ, sprite_codec_loss
from forge.sprite_latent.corpus import (
    FROZEN_PRODUCTION_CORPUS_SHA256,
    FROZEN_PRODUCTION_LEGAL_TUPLE_FINGERPRINT,
    FROZEN_PRODUCTION_SPLIT_FINGERPRINT,
)
from forge.sprite_latent.training import canonical_state_hash
from forge.sprite_latent_production import ProductionConfig, production_source_hash
from forge.sprite_latent_production.checkpoint import load_checkpoint, save_checkpoint_new, validate_checkpoint
from forge.sprite_latent_production.evaluation import evaluate_model
from forge.sprite_latent_production.loss import deterministic_sprite_codec_loss


def _tiny_config() -> ProductionConfig:
    return ProductionConfig(
        epochs=2,
        segment_epochs=1,
        batch_size=8,
        evaluation_batch_size=8,
        learning_rate=1.0e-3,
        minimum_learning_rate=1.0e-4,
        warmup_steps=0,
        continuous_warmup_epochs=1,
        seed=1234,
        width=8,
        residual_depth=1,
        condition_dim=16,
        latent_levels=(4, 4),
        worker_timeout_seconds=60,
    )


def _checkpoint_payload(config: ProductionConfig) -> dict[str, object]:
    model = SemanticSpriteFSQ(config.codec_config())
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    state = {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}
    return {
        "format": "nullvector-semantic-sprite-fsq-production-checkpoint-v1",
        "source_sha256": production_source_hash(),
        "corpus_sha256": FROZEN_PRODUCTION_CORPUS_SHA256,
        "split_fingerprint": FROZEN_PRODUCTION_SPLIT_FINGERPRINT,
        "legal_tuple_fingerprint": FROZEN_PRODUCTION_LEGAL_TUPLE_FINGERPRINT,
        "config": config.metadata(),
        "epoch": 0,
        "global_step": 0,
        "model_state": state,
        "ema_state": deepcopy(state),
        "optimizer_state": optimizer.state_dict(),
        "model_state_sha256": canonical_state_hash(model),
        "ema_state_sha256": canonical_state_hash(model),
        "history": [],
        "partial_epoch": None,
        "previous_checkpoint_sha256": None,
        "rng": {"torch": torch.get_rng_state(), "cuda": []},
    }


def test_production_config_roundtrips_exact_codec_contract() -> None:
    config = _tiny_config()
    assert ProductionConfig.from_metadata(config.metadata()) == config
    tampered = config.metadata()
    tampered["codec"]["implicit_code_count"] += 1
    with pytest.raises(ValueError, match="codec metadata mismatch"):
        ProductionConfig.from_metadata(tampered)
    with pytest.raises(ValueError, match="key mismatch"):
        ProductionConfig.from_metadata({**config.metadata(), "surprise": True})


def test_production_source_hash_is_complete_and_stable() -> None:
    first = production_source_hash()
    second = production_source_hash()
    assert first == second
    assert len(first) == 64
    assert set(first) <= set("0123456789abcdef")


def test_production_checkpoint_roundtrip_and_partial_epoch_contract(tmp_path: Path) -> None:
    config = _tiny_config()
    payload = _checkpoint_payload(config)
    path = tmp_path / "checkpoint.pt"
    digest = save_checkpoint_new(path, payload)
    assert len(digest) == 64
    loaded = load_checkpoint(path)
    assert loaded["model_state_sha256"] == payload["model_state_sha256"]
    assert loaded["partial_epoch"] is None
    with pytest.raises(FileExistsError):
        save_checkpoint_new(path, payload)

    partial = deepcopy(payload)
    partial["global_step"] = 7
    partial["partial_epoch"] = {"epoch": 1, "complete": False, "steps": 7}
    assert validate_checkpoint(partial)["global_step"] == 7
    partial["partial_epoch"]["complete"] = True
    with pytest.raises(ValueError, match="partial epoch"):
        validate_checkpoint(partial)


def test_production_checkpoint_rejects_resealed_authority_and_history_tamper() -> None:
    payload = _checkpoint_payload(_tiny_config())
    forged = deepcopy(payload)
    forged["corpus_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="frozen production authority"):
        validate_checkpoint(forged)

    forged = deepcopy(payload)
    forged["epoch"] = 1
    forged["global_step"] = 2
    forged["history"] = [{"epoch": 1, "complete": True, "steps": 1}]
    with pytest.raises(ValueError, match="global step"):
        validate_checkpoint(forged)

    forged = deepcopy(payload)
    first = next(iter(forged["model_state"]))
    forged["model_state"][first].reshape(-1)[0] = float("nan")
    model = SemanticSpriteFSQ(_tiny_config().codec_config())
    model.load_state_dict(forged["model_state"])
    forged["model_state_sha256"] = canonical_state_hash(model)
    with pytest.raises(ValueError, match="non-finite tensors"):
        validate_checkpoint(forged)


def test_background_only_codec_cannot_pass_foreground_quality_gates() -> None:
    config = _tiny_config()
    model = SemanticSpriteFSQ(config.codec_config())
    for parameter in model.parameters():
        parameter.data.zero_()
    count = 10
    part = np.zeros((count, 48, 48), dtype=np.uint8)
    material = np.zeros_like(part)
    emission = np.zeros_like(part)
    part[:, 18:30, 20:28] = 1
    corpus = SimpleNamespace(
        part_owner=part,
        material=material,
        emission_level=emission,
        genes=np.zeros((count, 24), dtype=np.float32),
        morphologies=np.asarray([index % 5 for index in range(count)], dtype=np.uint8),
        subtypes=np.asarray([(index % 5) * 4 for index in range(count)], dtype=np.uint8),
        roles=np.zeros((count,), dtype=np.uint8),
    )
    legal = torch.tensor(sorted(allowed_training_field_tuples()), dtype=torch.long)
    result = evaluate_model(
        model,
        corpus,
        np.arange(count, dtype=np.int64),
        legal,
        batch_size=8,
        device=torch.device("cpu"),
    )
    assert result["quality_accepted"] is False
    assert result["visible_tuple_accuracy"] == 0.0
    assert result["visible_silhouette_iou"] == 0.0
    assert result["quality_gates"]["visible_tuple_accuracy"] is False
    assert result["quality_gates"]["visible_silhouette_iou"] is False


def test_flattened_production_loss_matches_core_loss_on_cpu() -> None:
    torch.manual_seed(77)
    config = _tiny_config().codec_config()
    model = SemanticSpriteFSQ(config)
    count = 2
    part = torch.zeros((count, 48, 48), dtype=torch.long)
    material = torch.zeros_like(part)
    emission = torch.zeros_like(part)
    morphology = torch.tensor([0, 1])
    subtype = torch.tensor([0, 4])
    role = torch.tensor([0, 1])
    genes = torch.zeros((count, 24), dtype=torch.float32)
    output = model(part, material, emission, morphology, subtype, role, genes, quantize=True)
    legal = torch.tensor(sorted(allowed_training_field_tuples()), dtype=torch.long)
    core, core_parts = sprite_codec_loss(output, part, material, emission, legal, config=config)
    production, production_parts = deterministic_sprite_codec_loss(output, part, material, emission, legal, config=config)
    assert torch.allclose(core, production, rtol=1e-6, atol=1e-6)
    assert set(core_parts) == set(production_parts)
    for name in core_parts:
        assert torch.allclose(core_parts[name], production_parts[name], rtol=1e-6, atol=1e-6)
