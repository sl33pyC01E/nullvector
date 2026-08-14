from __future__ import annotations

from pathlib import Path

import pytest
import torch

from forge.organism_latent_flow.artifacts import validate_output
from forge.organism_latent_flow.contract import OrganismFlowConfig, source_manifest, source_sha256
from forge.organism_latent_flow.corpus import build_latent_corpus, load_latent_corpus, save_latent_corpus
from forge.organism_latent_flow.model import HierarchicalOrganismFlow, flow_matching_loss, integrate_flow
from forge.organism_latent_flow.training import checkpoint_name


def tiny_config() -> OrganismFlowConfig:
    return OrganismFlowConfig(coarse_width=64, fine_width=48, condition_dim=64, time_dim=32, depth=1)


@pytest.fixture(scope="module")
def latent_corpus() -> dict:
    return build_latent_corpus()


def test_flow_contract_and_capacity() -> None:
    assert len(source_sha256()) == 64 and len(source_manifest()) == 7
    model = HierarchicalOrganismFlow()
    assert sum(parameter.numel() for parameter in model.parameters()) == 13_661_120
    assert checkpoint_name(512) == "segment_0000512.pt"
    with pytest.raises(ValueError, match="latent pyramid"):
        OrganismFlowConfig(coarse_channels=16)


def test_latent_corpus_is_exactly_bound_and_roundtrips(tmp_path: Path, latent_corpus: dict) -> None:
    assert latent_corpus["semantic"]["sample_count"] == 45
    assert latent_corpus["semantic"]["family_census"] == [11, 10, 9, 8, 7]
    assert latent_corpus["tensors"]["coarse_mean"].shape == (45, 32, 12, 12)
    assert latent_corpus["tensors"]["fine_mean"].shape == (45, 16, 24, 24)
    path = tmp_path / "latent.pt"; save_latent_corpus(path, latent_corpus); loaded = load_latent_corpus(path)
    assert loaded["semantic"] == latent_corpus["semantic"]
    for key in loaded["tensors"]:
        assert torch.equal(loaded["tensors"][key], latent_corpus["tensors"][key])


def test_coupled_streams_and_condition_receive_gradients() -> None:
    model = HierarchicalOrganismFlow(tiny_config())
    generator = torch.Generator().manual_seed(7); batch = 2
    coarse_target = torch.randn((batch, 32, 12, 12), generator=generator)
    fine_target = torch.randn((batch, 16, 24, 24), generator=generator)
    coarse_noise = torch.randn(coarse_target.shape, generator=generator); fine_noise = torch.randn(fine_target.shape, generator=generator)
    condition = torch.randn((batch, 192), generator=generator); time = torch.tensor((.2, .8)); keep = torch.tensor((True, False))
    loss, pieces = flow_matching_loss(model, coarse_target, fine_target, condition, time, coarse_noise, fine_noise, keep); loss.backward()
    assert torch.isfinite(loss) and set(pieces) == {"loss", "coarse_mse", "fine_mse", "endpoint_l1"}
    for parameter in (model.condition[0].weight, model.coarse_in.weight, model.fine_in.weight, model.coarse_out[-1].weight, model.fine_out[-1].weight):
        assert parameter.grad is not None and float(parameter.grad.abs().sum()) > 0


def test_integration_is_deterministic_continuous_and_condition_sensitive() -> None:
    torch.manual_seed(13); model = HierarchicalOrganismFlow(tiny_config()).eval()
    coarse = torch.randn((2, 32, 12, 12)); fine = torch.randn((2, 16, 24, 24)); condition = torch.randn((2, 192))
    first = integrate_flow(model, coarse, fine, condition, steps=4, guidance=1.4)
    second = integrate_flow(model, coarse, fine, condition, steps=4, guidance=1.4)
    assert torch.equal(first[0], second[0]) and torch.equal(first[1], second[1])
    changed = integrate_flow(model, coarse, fine, condition.flip(0), steps=4, guidance=1.4)
    assert not torch.equal(first[0], changed[0]) and not torch.equal(first[1], changed[1])
    assert torch.isfinite(first[0]).all() and torch.isfinite(first[1]).all()


def test_frozen_flow_output_exact_replay_when_present() -> None:
    output = Path("outputs/organism_latent_flow/prior_v1")
    if not (output / "organism_flow_manifest.json").exists():
        pytest.skip("Frozen organism flow output is not present in this checkout")
    report = validate_output(output)
    assert report["status"] == "passed"
    assert report["claim_boundary"]["production_promotion_allowed"] is False
