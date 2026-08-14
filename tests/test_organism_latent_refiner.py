from __future__ import annotations

from pathlib import Path

import pytest
import torch

from forge.organism_latent_refiner.artifacts import _component_sizes, _metrics, validate_output
from forge.organism_latent_refiner.contract import OrganismRefinerConfig, source_manifest, source_sha256
from forge.organism_latent_refiner.model import HierarchicalLatentRefiner, latent_refinement_loss, refine_latents
from forge.organism_latent_refiner.training import _flow_authority, checkpoint_name


def tiny_config() -> OrganismRefinerConfig:
    return OrganismRefinerConfig(coarse_width=64, fine_width=64, condition_dim=64, time_dim=32, depth=1, auxiliary_batch=2)


def test_refiner_contract_is_bound_to_frozen_flow() -> None:
    assert len(source_sha256()) == 64 and len(source_manifest()) == 6
    checkpoint, contract = _flow_authority()
    assert checkpoint["step"] == 8192 and contract["total_steps"] == 8192
    assert checkpoint_name(512) == "segment_0000512.pt"
    model = HierarchicalLatentRefiner()
    assert sum(parameter.numel() for parameter in model.parameters()) == 4_894_096
    with pytest.raises(ValueError, match="scalar bounds"):
        OrganismRefinerConfig(corruption_min=.8, corruption_max=.2)


def test_near_zero_initialized_refiner_begins_near_identity() -> None:
    torch.manual_seed(11); model = HierarchicalLatentRefiner(tiny_config()).eval(); coarse = torch.randn((2, 32, 12, 12)); fine = torch.randn((2, 16, 24, 24)); condition = torch.randn((2, 192))
    refined = refine_latents(model, coarse, fine, condition)
    assert float((refined[0] - coarse).abs().max()) < .01 and float((refined[1] - fine).abs().max()) < .01


def test_refiner_streams_and_condition_receive_gradients() -> None:
    torch.manual_seed(17); model = HierarchicalLatentRefiner(tiny_config()); clean_coarse = torch.randn((2, 32, 12, 12)); clean_fine = torch.randn((2, 16, 24, 24)); corrupted_coarse = clean_coarse + .3 * torch.randn_like(clean_coarse); corrupted_fine = clean_fine + .3 * torch.randn_like(clean_fine); sigma = torch.tensor((.3, .3)); condition = torch.randn((2, 192))
    coarse_delta, fine_delta = model(corrupted_coarse, corrupted_fine, sigma, condition); loss, pieces, predicted_coarse, predicted_fine = latent_refinement_loss(coarse_delta, fine_delta, corrupted_coarse, corrupted_fine, clean_coarse, clean_fine); loss.backward()
    assert torch.isfinite(loss) and set(pieces) == {"latent_loss", "coarse_mse", "fine_mse", "coarse_l1", "fine_l1"}
    assert predicted_coarse.shape == clean_coarse.shape and predicted_fine.shape == clean_fine.shape
    for parameter in (model.condition[0].weight, model.coarse_in.weight, model.fine_in.weight, model.coarse_out[-1].weight, model.fine_out[-1].weight):
        assert parameter.grad is not None and float(parameter.grad.abs().sum()) > 0


def test_topology_metrics_measure_satellite_cells() -> None:
    rgba = torch.zeros((30, 4, 48, 48)); alpha = torch.zeros((30, 48, 48)); roles = torch.zeros((30, 8, 48, 48), dtype=torch.long); reference = torch.zeros((45, 4, 48, 48))
    alpha[0, 10:20, 10:20] = 1; alpha[0, 2, 2] = 1; alpha[1, 10:20, 10:20] = 1; rgba[:, 3] = alpha; roles[:, 0, 12, 12] = 1
    metrics, records = _metrics(rgba, alpha, roles, reference)
    assert records[0]["components"] == 2 and records[0]["satellite_cell_fraction"] == pytest.approx(1 / 101)
    assert records[1]["components"] == 1 and records[1]["satellite_cell_fraction"] == 0
    assert metrics["single_component_fraction"] == pytest.approx(1 / 30) and metrics["system_core_sample_fraction"] == pytest.approx(1 / 15)
    with pytest.raises(ValueError):
        _component_sizes(torch.zeros((1, 48, 48), dtype=torch.bool).numpy())


def test_frozen_refiner_exact_replay_when_present() -> None:
    output = Path("outputs/organism_latent_refiner/refiner_v3")
    if not (output / "organism_refiner_manifest.json").exists(): pytest.skip("Frozen organism refiner output is not present in this checkout")
    report = validate_output(output); assert report["status"] == "passed" and report["claim_boundary"]["production_promotion_allowed"] is False
