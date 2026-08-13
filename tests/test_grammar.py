from __future__ import annotations

import numpy as np

from forge.config import ARCHETYPES, LAYER_NAMES
from forge.grammar import (
    compose_rgba,
    genome_from_seed,
    genome_vector,
    layers_to_tokens,
    render_layers,
    tokens_to_layers,
)


def test_grammar_is_deterministic_and_nonempty() -> None:
    for archetype in range(len(ARCHETYPES)):
        genome = genome_from_seed(12_345, archetype)
        first = render_layers(genome)
        second = render_layers(genome)
        assert np.array_equal(first, second)
        assert first.shape == (len(LAYER_NAMES), 32, 32)
        assert first.dtype == np.uint8
        assert first[0].sum() >= 20
        assert first[5].sum() > 0
        assert np.isin(first, (0, 1)).all()


def test_genome_condition_vector_is_bounded() -> None:
    genes = genome_vector(genome_from_seed(99, "oracle"))
    assert genes.shape == (8,)
    assert np.isfinite(genes).all()
    assert (genes >= 0.0).all()
    assert (genes <= 1.0).all()


def test_token_contract_and_rgba_are_pixel_hard() -> None:
    layers = render_layers(genome_from_seed(777, "bulwark"))
    tokens = layers_to_tokens(layers)
    decoded = tokens_to_layers(tokens)
    rgba = compose_rgba(decoded, 777)
    assert tokens.shape == (32, 32)
    assert int(tokens.max()) <= len(LAYER_NAMES)
    assert rgba.shape == (32, 32, 4)
    assert set(np.unique(rgba[..., 3]).tolist()).issubset({0, 255})
