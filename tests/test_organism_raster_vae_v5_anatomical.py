from __future__ import annotations

import torch

from forge.organism_raster_vae_v3.contract import RasterVAEV3Config
from forge.organism_raster_vae_v5_anatomical.contract import (
    MAX_TOKENS,
    TOKEN_FEATURES,
    TOKEN_JOINT,
    TOKEN_ORGAN,
)
from forge.organism_raster_vae_v5_anatomical.dataset import AnatomicalGraphCorpus
from forge.organism_raster_vae_v5_anatomical.model import AnatomicalGraphRasterVAE, loss


def test_anatomical_token_and_authority_contract() -> None:
    corpus = AnatomicalGraphCorpus()
    families = set()
    for index in (0, 96, 192, 288, 384):
        row = corpus[index]
        families.add(int(row["family"]))
        assert row["tokens"].shape == (MAX_TOKENS, TOKEN_FEATURES)
        assert row["token_mask"].dtype == torch.bool
        assert bool((row["token_group"] == TOKEN_JOINT).any())
        assert bool((row["token_group"] == TOKEN_ORGAN).any())
        assert int(row["appendage_owner"].max()) < MAX_TOKENS
        assert int(row["joint_owner"].max()) < MAX_TOKENS
        assert int(row["organ_owner"].max()) < MAX_TOKENS
    assert families == set(range(5))


def test_anatomical_forward_and_three_authority_losses() -> None:
    corpus = AnatomicalGraphCorpus()
    rows = [corpus[index] for index in (0, 96)]
    batch = {key: torch.stack([row[key] for row in rows]) for key in rows[0]}
    model = AnatomicalGraphRasterVAE(RasterVAEV3Config())
    output = model(
        batch["living"],
        batch["family"],
        batch["traits"],
        batch["phase"],
        batch["tokens"],
        batch["token_mask"],
        stochastic=False,
    )
    value, metrics = loss(output, batch, model.config, .1)
    assert output.attention12.shape == (2, 12 * 12, MAX_TOKENS)
    assert output.attention24.shape == (2, 24 * 24, MAX_TOKENS)
    assert torch.isfinite(value)
    assert metrics["appendage_owner_nll"] > 0
    assert metrics["joint_owner_nll"] > 0
    assert metrics["organ_owner_nll"] > 0
    assert metrics["attention_hierarchy"] >= 0
