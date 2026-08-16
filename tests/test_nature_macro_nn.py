from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from forge.nature_macro_nn.contract import GLOBAL_FEATURES, STATE_CHANNELS, ModelConfig
from forge.nature_macro_nn.corpus import build_corpus, generate_world_arrays, validate_corpus
from forge.nature_macro_nn.model import NeuralMacroPatchDynamics
from forge.nature_macro_nn.state import extract_global_state, extract_patch_state
from forge.nature_macro_nn.corpus import _bootstrap
from forge.nature_world_scale_v1.atlas import BIOMES


def test_state_projection_preserves_all_biomes_and_city_layers() -> None:
    vectors = []
    for index, biome in enumerate(BIOMES):
        world, society = _bootstrap(0xB10B1E + index, index)
        world.biome = biome
        patch = extract_patch_state(world, society)
        global_state = extract_global_state(world, society)
        assert patch.shape == (len(STATE_CHANNELS), 32, 32)
        assert global_state.shape == (GLOBAL_FEATURES,)
        assert np.isfinite(patch).all() and np.isfinite(global_state).all()
        assert np.all((patch >= 0) & (patch <= 1))
        assert global_state[36:].sum() == pytest.approx(1.0)
        vectors.append(tuple(global_state[36:]))
    assert len(set(vectors)) == len(BIOMES)


def test_zero_initialized_model_is_exact_persistence_with_trainable_heads() -> None:
    torch.manual_seed(7)
    model = NeuralMacroPatchDynamics(ModelConfig(width=32, blocks=2, global_width=48))
    current = torch.rand(2, len(STATE_CHANNELS), 32, 32)
    previous = torch.rand_like(current)
    global_state = torch.rand(2, GLOBAL_FEATURES)
    previous_global = torch.rand_like(global_state)
    predicted, predicted_global, *_ = model(current, previous, global_state, previous_global)
    torch.testing.assert_close(predicted, current, rtol=0, atol=0)
    torch.testing.assert_close(predicted_global, global_state, rtol=0, atol=0)
    (predicted.mean() + predicted_global.mean()).backward()
    assert model.delta.weight.grad is not None
    assert model.global_head[-1].weight.grad is not None


def test_world_sequence_and_corpus_round_trip(tmp_path: Path) -> None:
    arrays = generate_world_arrays(0, steps=8, base_seed=0xCAFE)
    assert arrays["current"].shape == (8, len(STATE_CHANNELS), 32, 32)
    assert arrays["global_state"].shape == (8, GLOBAL_FEATURES)
    assert float(np.abs(arrays["target"].astype(np.float32) - arrays["current"].astype(np.float32)).sum()) > 0
    root = tmp_path / "macro-corpus"
    built = build_corpus(root, worlds=1, steps=8, base_seed=0xCAFE)
    assert built["passed"] and built["pairs"] == 8
    assert validate_corpus(root) == built


def test_corpus_manifest_tamper_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "macro-corpus"
    build_corpus(root, worlds=1, steps=8, base_seed=0xBEEF)
    manifest = root / "manifest.json"
    payload = manifest.read_text("utf-8").replace('"pairs":8', '"pairs":9')
    manifest.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match="provenance"):
        validate_corpus(root)
