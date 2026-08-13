from __future__ import annotations

import torch

from forge.multifield_diffusion import (
    MultiFieldSpriteDiffusion,
    MultiFieldVocabulary,
    multifield_diffusion_loss,
    seeded_generators,
)


def _small_model(*, steps: int = 2) -> MultiFieldSpriteDiffusion:
    return MultiFieldSpriteDiffusion(
        vocabulary=MultiFieldVocabulary(11, 7, 4),
        morphology_count=5,
        subtype_count=15,
        role_count=6,
        gene_dim=24,
        guide_channels=8,
        steps=steps,
        width=32,
        image_size=16,
    )


def test_multifield_forward_corruption_and_loss_are_aligned() -> None:
    model = _small_model()
    batch = 2
    part = torch.randint(0, 11, (batch, 16, 16))
    material = torch.randint(0, 7, (batch, 16, 16))
    emission = torch.randint(0, 4, (batch, 16, 16))
    guide = torch.rand(batch, 8, 16, 16)
    morphologies = torch.tensor([0, 4])
    subtypes = torch.tensor([2, 13])
    roles = torch.tensor([1, 5])
    genes = torch.rand(batch, 24)
    timesteps = torch.tensor([1, 2])
    corrupted_part, corrupted_material, corrupted_emission, masked = model.corrupt(
        part, material, emission, timesteps
    )
    assert torch.equal(corrupted_part == model.part_mask_token, masked)
    assert torch.equal(corrupted_material == model.material_mask_token, masked)
    assert torch.equal(corrupted_emission == model.emission_mask_token, masked)
    logits = model(
        corrupted_part,
        corrupted_material,
        corrupted_emission,
        guide,
        morphologies,
        subtypes,
        roles,
        genes,
        timesteps,
    )
    result = multifield_diffusion_loss(
        logits, part, material, emission, masked
    )
    assert logits.part.shape == (batch, 11, 16, 16)
    assert logits.material.shape == (batch, 7, 16, 16)
    assert logits.emission.shape == (batch, 4, 16, 16)
    assert torch.isfinite(result.loss)
    assert 0.0 < float(result.masked_fraction) <= 1.0


def test_multifield_sampler_is_seed_deterministic_and_fully_resolves() -> None:
    torch.manual_seed(91)
    model = _small_model(steps=2).eval()
    guide = torch.rand(1, 8, 16, 16)
    morphology = torch.tensor([2])
    subtype = torch.tensor([7])
    role = torch.tensor([3])
    genes = torch.rand(1, 24)

    def sample(seed: int):
        generator = torch.Generator().manual_seed(seed)
        return model.sample(
            guide,
            morphology,
            subtype,
            role,
            genes,
            generators=[generator],
        )

    first = sample(888)
    second = sample(888)
    assert all(torch.equal(left, right) for left, right in zip(first, second))
    part, material, emission = first
    assert int(part.max()) < model.vocabulary.part_count
    assert int(material.max()) < model.vocabulary.material_count
    assert int(emission.max()) < model.vocabulary.emission_count
    assert model.architecture_config()["image_size"] == 16


def test_multifield_corruption_zero_is_identity_and_explicit_rng_replays() -> None:
    model = _small_model()
    shape = (2, 16, 16)
    part = torch.randint(0, 11, shape)
    material = torch.randint(0, 7, shape)
    emission = torch.randint(0, 4, shape)
    zero = model.corrupt(
        part,
        material,
        emission,
        torch.zeros(2, dtype=torch.long),
        generator=torch.Generator().manual_seed(4),
    )
    assert not zero[3].any()
    assert torch.equal(zero[0], part)
    first = model.corrupt(
        part,
        material,
        emission,
        torch.tensor([1, 2]),
        generator=torch.Generator().manual_seed(55),
    )
    second = model.corrupt(
        part,
        material,
        emission,
        torch.tensor([1, 2]),
        generator=torch.Generator().manual_seed(55),
    )
    assert all(torch.equal(left, right) for left, right in zip(first, second))
    unmasked = ~first[3]
    assert torch.equal(first[0][unmasked], part[unmasked])
    assert torch.equal(first[1][unmasked], material[unmasked])
    assert torch.equal(first[2][unmasked], emission[unmasked])


def test_forced_single_mask_uses_explicit_generator_and_seed_helper() -> None:
    model = MultiFieldSpriteDiffusion(
        vocabulary=MultiFieldVocabulary(11, 7, 4),
        steps=10_000,
        width=32,
        image_size=16,
    )
    shape = (2, 16, 16)
    targets = [torch.zeros(shape, dtype=torch.long) for _ in range(3)]
    timestep = torch.ones(2, dtype=torch.long)
    first = model.corrupt(
        *targets,
        timestep,
        generator=seeded_generators([123], "cpu")[0],
    )
    second = model.corrupt(
        *targets,
        timestep,
        generator=seeded_generators([123], torch.device("cpu"))[0],
    )
    assert all(torch.equal(left, right) for left, right in zip(first, second))
    assert first[3].flatten(1).sum(dim=1).tolist() == [1, 1]


def test_multifield_rejects_unsafe_widths_and_loss_weights() -> None:
    import pytest

    with pytest.raises(ValueError, match="thirty-two"):
        MultiFieldSpriteDiffusion(width=48)
    model = _small_model()
    batch = 1
    targets = [torch.zeros(batch, 16, 16, dtype=torch.long) for _ in range(3)]
    guide = torch.zeros(batch, 8, 16, 16)
    logits = model(
        targets[0], targets[1], targets[2], guide,
        torch.zeros(batch, dtype=torch.long),
        torch.zeros(batch, dtype=torch.long),
        torch.zeros(batch, dtype=torch.long),
        torch.zeros(batch, 24),
        torch.ones(batch, dtype=torch.long),
    )
    with pytest.raises(ValueError, match="positive sum"):
        multifield_diffusion_loss(
            logits, *targets, torch.ones(batch, 16, 16, dtype=torch.bool),
            field_weights=(0.0, 0.0, 0.0),
        )


def test_multifield_sampler_respects_legal_joint_tuples() -> None:
    model = _small_model(steps=2).eval()
    legal = torch.tensor([[0, 0, 0], [1, 2, 0], [4, 6, 3]], dtype=torch.long)
    result = model.sample(
        torch.zeros(1, 8, 16, 16),
        torch.tensor([1]),
        torch.tensor([2]),
        torch.tensor([3]),
        torch.rand(1, 24),
        generators=[torch.Generator().manual_seed(22)],
        legal_tuples=legal,
    )
    triples = torch.stack(result, dim=-1).reshape(-1, 3)
    assert all(any(torch.equal(row, allowed) for allowed in legal) for row in triples)
