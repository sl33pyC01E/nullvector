from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from forge.map_topology_neural_prior_v2.conditioning import CONDITION_CHANNELS, build_spatial_conditions
from forge.map_topology_neural_prior_v2.contract import PriorV2Config, canonical_json_bytes, prior_v2_source_sha256
from forge.map_topology_neural_prior_v2.masking import MASK_MODES_V2, mask_tokens_v2
from forge.map_topology_neural_prior_v2.model import build_prior_v2, masked_token_loss_v2, sample_parallel_v2
from forge.map_topology_neural_prior_v2.smoke import build_smoke, validate_smoke


CORPUS = Path("outputs/map_decorator_corpus_v1")
LATENTS = Path("outputs/map_topology_neural_prior_corpus/v1")


def _batch(height: int = 13, width: int = 19, count: int = 6) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(1234)
    points = torch.zeros((count, 4, height, width), dtype=torch.float32)
    points[:, 0, 1, 1] = 1
    points[:, 1, -2, -2] = 1
    points[:, 2, height // 2, width // 2] = 1
    points[:, 3, -3, 2] = 1
    return {
        "targets": torch.randint(0, 512, (count, height, width), generator=generator),
        "valid_mask": torch.ones((count, 1, height, width), dtype=torch.bool),
        "point_conditions": points,
        "global_conditions": torch.rand((count, 14), generator=generator),
        "theme_index": torch.arange(count, dtype=torch.long),
    }


def test_v2_contract_is_additive_and_bounded() -> None:
    assert len(prior_v2_source_sha256()) == 64
    config = PriorV2Config()
    assert config.levels == 3
    with pytest.raises(ValueError, match="levels"):
        PriorV2Config(levels=1)
    with pytest.raises(ValueError, match="mask"):
        PriorV2Config(minimum_mask_fraction=1.0)


def test_spatial_conditioning_has_exact_channels_and_truthful_corridor() -> None:
    batch = _batch(count=2)
    first = build_spatial_conditions(batch["point_conditions"], batch["valid_mask"])
    second = build_spatial_conditions(batch["point_conditions"], batch["valid_mask"])
    assert first.shape == (2, len(CONDITION_CHANNELS), 13, 19)
    assert torch.equal(first, second)
    corridor = first[:, CONDITION_CHANNELS.index("mission_corridor")]
    assert bool((corridor >= 0).all()) and bool((corridor <= 1).all())
    assert float(corridor[:, 1, 1].mean()) > float(corridor[:, 1, -2].mean())
    broken = batch["point_conditions"].clone()
    broken[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        build_spatial_conditions(broken, batch["valid_mask"])


def test_full_mask_is_a_real_training_mode_and_all_modes_replay() -> None:
    batch = _batch()
    config = PriorV2Config()
    first = mask_tokens_v2(batch["targets"], batch["valid_mask"], generator=torch.Generator().manual_seed(77), config=config, step=0)
    second = mask_tokens_v2(batch["targets"], batch["valid_mask"], generator=torch.Generator().manual_seed(77), config=config, step=0)
    assert first["modes"] == list(MASK_MODES_V2)
    assert torch.equal(first["mask"], second["mask"])
    assert bool(first["mask"][0].all())
    assert first["mask_fraction"][0, 0] == 1
    for index in range(1, 6):
        assert 0 < int(first["mask"][index].sum()) < int(batch["valid_mask"][index].sum())


def test_multiscale_model_handles_odd_rectangles_and_has_global_influence() -> None:
    batch = _batch(height=31, width=47, count=1)
    config = PriorV2Config(width=8, levels=3, steps=1)
    tokens = torch.full((1, 31, 47), 512, dtype=torch.long)
    model = build_prior_v2(config)
    inputs = {
        "tokens": tokens, "valid_mask": batch["valid_mask"], "point_conditions": batch["point_conditions"],
        "global_conditions": batch["global_conditions"], "theme_index": batch["theme_index"],
        "mask_fraction": torch.ones((1, 1)),
    }
    with torch.inference_mode(), torch.backends.mkldnn.flags(enabled=False):
        baseline = model(inputs)
        changed_tokens = tokens.clone()
        changed_tokens[0, 0, 0] = 5
        changed = model({**inputs, "tokens": changed_tokens})
        theme_changed = model({**inputs, "theme_index": torch.ones(1, dtype=torch.long)})
    assert baseline.shape == (1, 512, 31, 47)
    assert float((baseline[0, :, -1, -1] - changed[0, :, -1, -1]).abs().max()) > 1e-8
    assert float((baseline - theme_changed).abs().max()) > 1e-8


def test_v2_loss_and_full_generation_are_finite() -> None:
    batch = _batch(height=9, width=11, count=2)
    config = PriorV2Config(width=8, levels=2, steps=1, sampling_steps=3)
    masked = mask_tokens_v2(batch["targets"], batch["valid_mask"], generator=torch.Generator().manual_seed(9), config=config, step=0)
    model = build_prior_v2(config)
    inputs = {
        "tokens": masked["tokens"], "valid_mask": batch["valid_mask"], "point_conditions": batch["point_conditions"],
        "global_conditions": batch["global_conditions"], "theme_index": batch["theme_index"], "mask_fraction": masked["mask_fraction"],
    }
    logits = model(inputs)
    loss = masked_token_loss_v2(logits, batch["targets"], masked["mask"])
    loss.backward()
    assert bool(torch.isfinite(loss))
    sampled = sample_parallel_v2(model.eval(), {name: inputs[name] for name in ("valid_mask", "point_conditions", "global_conditions", "theme_index")}, sampling_steps=3)
    assert sampled["tokens"].shape == batch["targets"].shape
    assert bool(((sampled["tokens"] >= 0) & (sampled["tokens"] < 512)).all())


def test_v2_smoke_exact_replay_and_fail_closed_tamper(tmp_path: Path) -> None:
    output = tmp_path / "prior-v2-smoke"
    built = build_smoke(output, corpus_root=CORPUS, latent_root=LATENTS)
    assert built == validate_smoke(output, corpus_root=CORPUS, latent_root=LATENTS)
    assert built["status"] == "passed"
    assert built["gates"]["full_mask_training_exercised"] is True
    assert built["metrics"]["sample_count"] == 6
    manifest_path = output / "smoke_manifest.json"
    original = manifest_path.read_bytes()
    try:
        payload = json.loads(original)
        payload["gates"]["invented"] = True
        payload.pop("manifest_sha256")
        payload["manifest_sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
        manifest_path.write_bytes(canonical_json_bytes(payload))
        with pytest.raises(ValueError, match="gates"):
            validate_smoke(output, corpus_root=CORPUS, latent_root=LATENTS)
    finally:
        manifest_path.write_bytes(original)
