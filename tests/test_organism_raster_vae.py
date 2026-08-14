from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch.utils.data._utils.collate import default_collate

from forge.organism_raster_vae.contract import FROZEN_UPSTREAM, OrganismVAEConfig, organism_vae_source_sha256, source_manifest
from forge.organism_raster_vae.dataset import GENE_NAMES, OrganismRasterCorpus
from forge.organism_raster_vae.model import ContinuousOrganismRasterVAE, organism_vae_loss
from forge.organism_raster_vae.smoke import run_smoke, validate_smoke


@pytest.fixture(scope="module")
def corpus() -> OrganismRasterCorpus:
    return OrganismRasterCorpus()


def test_corpus_is_exact_cellular_living_field(corpus: OrganismRasterCorpus) -> None:
    assert len(corpus) == 45
    assert [len(corpus.indices_by_family[family]) for family in range(5)] == [11, 10, 9, 8, 7]
    assert len(GENE_NAMES) == 16 and set(FROZEN_UPSTREAM) == {"anatomy_manifest_sha256", "physiology_manifest_sha256", "trauma_manifest_sha256"}
    for index in range(len(corpus)):
        sample = corpus[index]
        assert sample["living_field"].shape == (74, 48, 48)
        assert sample["rgba"].shape == (4, 48, 48)
        assert sample["physiology"].shape == (8, 48, 48)
        assert sample["cell_state"].shape == (10, 48, 48)
        assert torch.isfinite(sample["living_field"]).all()
        assert torch.equal(sample["living_field"][0], sample["occupancy"])
        assert torch.equal(sample["rgba"][3], sample["occupancy"])


def test_continuous_latent_rasterizer_has_reconstruction_gradients(corpus: OrganismRasterCorpus) -> None:
    config = OrganismVAEConfig(width=16, latent_channels=4, residual_depth=1)
    model = ContinuousOrganismRasterVAE(config)
    batch = default_collate([corpus[index] for index in range(2)])
    output = model(batch["living_field"], batch["family"], batch["subtype"], batch["role"], batch["genes"], generator=torch.Generator().manual_seed(7))
    assert output.mean.shape == output.log_variance.shape == output.latent.shape == (2, 4, 12, 12)
    assert output.rgba.shape == (2, 4, 48, 48) and output.physiology.shape == (2, 8, 48, 48)
    loss, metrics = organism_vae_loss(output, batch, config)
    loss.backward()
    assert set(metrics) == {"loss", "reconstruction", "occupancy_bce", "rgba_l1", "categorical_ce", "emission_l1", "physiology_l1", "cell_state_l1", "kl", "symmetry_l1", "alpha_consistency_l1", "edge_l1"}
    assert torch.isfinite(loss) and float(model.mean.weight.grad.abs().sum()) > 0 and float(model.rgba_head.weight.grad.abs().sum()) > 0


def test_latent_fusion_and_mutation_are_continuous(corpus: OrganismRasterCorpus) -> None:
    config = OrganismVAEConfig(width=16, latent_channels=4, residual_depth=1); model = ContinuousOrganismRasterVAE(config).eval()
    indices = corpus.indices_by_family[1][:2]; batch = default_collate([corpus[index] for index in indices])
    with torch.inference_mode():
        condition = model.condition_vector(batch["family"], batch["subtype"], batch["role"], batch["genes"]); mean, _ = model.encode(batch["living_field"], condition)
        left = model.decode(mean[:1], condition[:1]).rgba; right = model.decode(mean[1:], condition[1:]).rgba
        middle = model.decode(mean[:1] * .5 + mean[1:] * .5, condition[:1] * .5 + condition[1:] * .5).rgba
        mutation = model.decode(mean[:1] + .25, condition[:1]).rgba
    assert not torch.equal(left, right) and not torch.equal(middle, left) and not torch.equal(middle, right)
    assert not torch.equal(mutation, left) and bool(((middle > 0) & (middle < 1)).all())


def test_cpu_smoke_exact_replay_and_tamper_rejection(tmp_path: Path) -> None:
    output = tmp_path / "smoke"; report = run_smoke(output, device_name="cpu", steps=4)
    assert report["status"] == "passed" and report["claim_boundary"]["production_promotion_allowed"] is False
    assert report["source_sha256"] == organism_vae_source_sha256() and report["source_manifest"] == source_manifest()
    assert validate_smoke(output) == report
    artifact = output / report["artifacts"]["fusion"]["path"]; original = artifact.read_bytes()
    try:
        artifact.write_bytes(original + b"tamper")
        with pytest.raises(ValueError, match="artifact identity"):
            validate_smoke(output)
    finally:
        artifact.write_bytes(original)
    manifest_path = output / "organism_vae_manifest.json"; payload = json.loads(manifest_path.read_text())
    payload["metrics"]["silhouette_iou"] = 1.0
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical JSON|self-hash"):
        validate_smoke(output)
