from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from forge.map_topology_neural_prior.contract import (
    CODEBOOK_SIZE,
    FROZEN_CODEC_CHECKPOINT_SHA256,
    MASK_TOKEN,
    MaskedPriorConfig,
    canonical_json_bytes,
    prior_source_sha256,
    sha256_file,
)
from forge.map_topology_neural_prior.dataset import FrozenLatentDataset
from forge.map_topology_neural_prior.masking import MASK_MODES, mask_tokens
from forge.map_topology_neural_prior.model import build_prior, masked_token_loss
from forge.map_topology_neural_prior.smoke import build_smoke, validate_smoke


CORPUS = Path("outputs/map_decorator_corpus_v1")


@pytest.fixture(scope="module")
def latent_dataset() -> FrozenLatentDataset:
    return FrozenLatentDataset(CORPUS)


@pytest.fixture(scope="module")
def latent_batch(latent_dataset: FrozenLatentDataset):
    return latent_dataset.encode(latent_dataset.smoke_refs())


def test_masked_prior_contract_binds_accepted_codec_and_stays_cpu_only(
    latent_dataset: FrozenLatentDataset,
) -> None:
    assert len(prior_source_sha256()) == 64
    assert sha256_file(latent_dataset.checkpoint_path) == FROZEN_CODEC_CHECKPOINT_SHA256
    assert not torch.cuda.is_initialized()
    assert all(not parameter.requires_grad for parameter in latent_dataset.codec.parameters())
    with pytest.raises(ValueError, match="one or two"):
        MaskedPriorConfig(steps=3)


def test_masked_prior_latent_encoding_is_exact_and_balanced_by_theme(
    latent_dataset: FrozenLatentDataset,
    latent_batch,
) -> None:
    batch, identity = latent_batch
    refs = latent_dataset.smoke_refs()
    assert {ref.theme for ref in refs} == {"arena", "rooms", "caves", "archipelago", "garden", "anomaly"}
    assert batch["targets"].shape == (6, 8, 8)
    assert batch["targets"].dtype == torch.long
    assert bool(((batch["targets"] >= 0) & (batch["targets"] < CODEBOOK_SIZE)).all())
    replay, replay_identity = latent_dataset.encode(refs)
    assert identity == replay_identity
    assert all(torch.equal(batch[name], replay[name]) for name in batch)


def test_masked_prior_structured_masks_are_deterministic_and_nontrivial(latent_batch) -> None:
    batch, _ = latent_batch
    config = MaskedPriorConfig()
    first = mask_tokens(
        batch["targets"], batch["valid_mask"],
        generator=torch.Generator().manual_seed(123), config=config, step=0,
    )
    second = mask_tokens(
        batch["targets"], batch["valid_mask"],
        generator=torch.Generator().manual_seed(123), config=config, step=0,
    )
    assert torch.equal(first["tokens"], second["tokens"])
    assert torch.equal(first["mask"], second["mask"])
    assert set(first["modes"]) == set(MASK_MODES)
    for index in range(batch["targets"].shape[0]):
        valid_count = int(batch["valid_mask"][index].sum())
        masked_count = int(first["mask"][index].sum())
        assert 0 < masked_count < valid_count
    assert bool((first["tokens"][first["mask"]] == MASK_TOKEN).all())
    assert torch.equal(first["tokens"][~first["mask"]], batch["targets"][~first["mask"]])


def test_masked_prior_loss_is_finite_and_updates_only_prior(latent_batch) -> None:
    batch, _ = latent_batch
    config = MaskedPriorConfig(width=8, residual_depth=0, steps=1)
    masked = mask_tokens(
        batch["targets"], batch["valid_mask"],
        generator=torch.Generator().manual_seed(456), config=config, step=1,
    )
    model = build_prior(config)
    logits = model({
        **{name: batch[name] for name in ("valid_mask", "point_conditions", "global_conditions", "theme_index")},
        "tokens": masked["tokens"],
        "mask_fraction": masked["mask_fraction"],
    })
    loss = masked_token_loss(logits, batch["targets"], masked["mask"])
    loss.backward()
    assert bool(torch.isfinite(loss))
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_masked_prior_smoke_exact_replay_and_tamper_rejection(tmp_path: Path) -> None:
    output = tmp_path / "prior-smoke"
    built = build_smoke(output, corpus_root=CORPUS)
    replay = validate_smoke(output, corpus_root=CORPUS)
    assert built == replay
    assert built["status"] == "passed"
    assert built["metrics"]["raw_sample_count"] == 6
    assert built["metrics"]["unique_raw_samples"] >= 2
    assert built["claim_boundary"]["generative_quality_claim"] is False
    raw_path = output / "raw_latent_bank.json"
    original = raw_path.read_bytes()
    try:
        raw = json.loads(original)
        raw["samples"][0]["tokens"][0][0] = (raw["samples"][0]["tokens"][0][0] + 1) % CODEBOOK_SIZE
        raw_path.write_text(json.dumps(raw, sort_keys=True), encoding="utf-8")
        with pytest.raises(ValueError, match="canonical JSON|file identity"):
            validate_smoke(output, corpus_root=CORPUS)
    finally:
        raw_path.write_bytes(original)
    manifest_path = output / "smoke_manifest.json"
    original_manifest = manifest_path.read_bytes()
    try:
        manifest = json.loads(original_manifest)
        manifest["gates"]["invented_gate"] = True
        manifest.pop("manifest_sha256")
        manifest["manifest_sha256"] = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()
        manifest_path.write_bytes(canonical_json_bytes(manifest))
        with pytest.raises(ValueError, match="safety gates"):
            validate_smoke(output, corpus_root=CORPUS)
    finally:
        manifest_path.write_bytes(original_manifest)
