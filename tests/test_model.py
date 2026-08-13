from __future__ import annotations

import torch

from forge.diffusion import (
    MASK_TOKEN,
    CategoricalSpriteDiffusion,
    categorical_diffusion_loss,
)
from forge.model import SemanticBetaVAE, vae_loss


def test_categorical_diffusion_shapes_and_masked_loss() -> None:
    model = CategoricalSpriteDiffusion(width=32)
    tokens = torch.randint(0, 9, (2, 32, 32))
    archetypes = torch.tensor([0, 3])
    genes = torch.rand(2, 8)
    timesteps = torch.tensor([3, 12])
    corrupted, masked = model.corrupt(tokens, timesteps)
    logits = model(corrupted, archetypes, genes, timesteps)
    result = categorical_diffusion_loss(logits, tokens, masked)
    assert logits.shape == (2, 9, 32, 32)
    assert (corrupted == MASK_TOKEN).any()
    assert torch.isfinite(result.loss)
    assert 0.0 <= float(result.accuracy) <= 1.0


def test_categorical_diffusion_supports_48px_morphology_models() -> None:
    model = CategoricalSpriteDiffusion(
        token_count=18,
        archetype_count=5,
        gene_dim=24,
        steps=2,
        width=32,
        image_size=48,
    )
    archetypes = torch.tensor([4])
    genes = torch.rand(1, 24)
    generator = torch.Generator().manual_seed(123)
    tokens = model.sample(archetypes, genes, generators=[generator])
    assert tokens.shape == (1, 48, 48)
    assert int(tokens.min()) >= 0
    assert int(tokens.max()) < 18
    assert model.architecture_config()["image_size"] == 48


def test_semantic_vae_is_a_real_stochastic_posterior() -> None:
    model = SemanticBetaVAE()
    layers = torch.rand(2, 8, 32, 32).round()
    labels = torch.tensor([1, 2])
    output = model(layers, labels)
    loss, pieces = vae_loss(
        output,
        layers,
        beta=1.0e-3,
        dice_weight=0.35,
        pos_weight=torch.ones(8),
    )
    assert output.logits.shape == layers.shape
    assert output.mu.shape == (2, 32)
    assert not torch.equal(output.latent, output.mu)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in pieces.values())
