from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from forge.map_topology_neural_production.dataset import TopologyProductionDataset
from forge.map_topology_neural_prior_generation.bank import (
    generate_case,
    plan_cases,
    validate_case,
    verify_case_result,
)
from forge.map_topology_neural_prior_generation.contract import GenerationConfig, canonical_json_bytes
from forge.map_topology_neural_prior_generation.sampling import sample_seeded_parallel


PROJECT = Path(__file__).resolve().parents[1]
CORPUS = PROJECT / "outputs" / "map_decorator_corpus_v1"


class CoordinatePrior(torch.nn.Module):
    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        tokens = batch["tokens"]
        _, height, width = tokens.shape
        y = torch.arange(height).view(1, 1, height, 1)
        x = torch.arange(width).view(1, 1, 1, width)
        vocabulary = torch.arange(512).view(1, 512, 1, 1)
        target = (x * 17 + y * 29 + 13) % 512
        return -torch.abs(vocabulary - target).float() * 0.07


def conditions(height: int = 8, width: int = 8) -> dict[str, torch.Tensor]:
    valid = torch.ones((1, 1, height, width), dtype=torch.bool)
    return {
        "valid_mask": valid,
        "point_conditions": torch.zeros((1, 4, height, width), dtype=torch.float32),
        "global_conditions": torch.zeros((1, 14), dtype=torch.float32),
        "theme_index": torch.zeros((1,), dtype=torch.long),
    }


def test_seeded_sampler_is_exact_and_seed_sensitive() -> None:
    model = CoordinatePrior()
    first = sample_seeded_parallel(model, conditions(), sampling_steps=8, seed=91, temperature=.8, top_k=16)
    replay = sample_seeded_parallel(model, conditions(), sampling_steps=8, seed=91, temperature=.8, top_k=16)
    alternate = sample_seeded_parallel(model, conditions(), sampling_steps=8, seed=92, temperature=.8, top_k=16)
    assert torch.equal(first.tokens, replay.tokens)
    assert torch.equal(first.uncertainty, replay.uncertainty)
    assert first.trace == replay.trace
    assert not torch.equal(first.tokens, alternate.tokens)
    assert len(first.trace) == 8
    assert sum(int(row["revealed"]) for row in first.trace) == 64
    assert not bool((first.tokens == 512).any())


@pytest.mark.parametrize(
    "kwargs",
    (
        {"variants_per_condition": 0}, {"sampling_steps": 1}, {"temperature": float("nan")},
        {"top_k": 513}, {"maximum_workers": 3}, {"maximum_attempts": 4},
        {"worker_timeout_seconds": 59},
    ),
)
def test_generation_config_fails_closed(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        GenerationConfig(**kwargs)


def test_case_plan_is_balanced_and_order_independent() -> None:
    dataset = TopologyProductionDataset(CORPUS)
    config = GenerationConfig(variants_per_condition=2)
    plan = plan_cases(dataset, config)
    assert len(plan) == 48
    assert len({item.case_id for item in plan}) == 48
    assert {item.ref.theme for item in plan} == {"arena", "rooms", "caves", "archipelago", "garden", "anomaly"}
    assert {item.ref.width for item in plan} == {32, 72, 128, 256}
    assert all(sum(item.ref.theme == theme for item in plan) == 8 for theme in {item.ref.theme for item in plan})
    assert plan == plan_cases(dataset, config)


@pytest.fixture(scope="module")
def real_case(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, GenerationConfig]:
    destination = tmp_path_factory.mktemp("prior-generation") / "case"
    config = GenerationConfig(variants_per_condition=1)
    generate_case(destination, corpus_root=CORPUS, config=config, case_id="anomaly_32x32_bee12f02_v00")
    return destination, config


def test_real_heldout_case_has_exact_neural_and_compiler_replay(real_case: tuple[Path, GenerationConfig]) -> None:
    destination, config = real_case
    result = verify_case_result(destination, corpus_root=CORPUS, config=config)
    assert result["exact"] is True
    manifest = validate_case(destination, corpus_root=CORPUS, config=config, exact_neural_replay=True)
    assert manifest["conditioning"]["fully_masked"] is True
    assert manifest["conditioning"]["target_latent_tokens_accessed"] is False
    assert manifest["gates"]["compiled_valid"] is True
    assert manifest["sampler"]["unique_tokens"] >= 1


def test_case_artifact_tamper_is_rejected(real_case: tuple[Path, GenerationConfig]) -> None:
    destination, config = real_case
    preview = destination / "preview.png"
    original = preview.read_bytes()
    try:
        preview.write_bytes(original + b"tamper")
        with pytest.raises(ValueError, match="artifact closure"):
            validate_case(destination, corpus_root=CORPUS, config=config, exact_neural_replay=False)
    finally:
        preview.write_bytes(original)


def test_case_manifest_semantic_tamper_is_rejected(real_case: tuple[Path, GenerationConfig]) -> None:
    destination, config = real_case
    path = destination / "case_manifest.json"
    original = path.read_bytes()
    try:
        payload = json.loads(original)
        payload["metrics"]["raw_openness"] = .99
        path.write_bytes(canonical_json_bytes(payload))
        with pytest.raises(ValueError, match="self-hash"):
            validate_case(destination, corpus_root=CORPUS, config=config, exact_neural_replay=False)
    finally:
        path.write_bytes(original)
