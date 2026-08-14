from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data._utils.collate import default_collate

from forge.organism_raster_vae_v2.contract import OrganismVAEV2Config, authority, source_manifest, source_sha256
from forge.organism_raster_vae_v2.dataset import OrganismRasterCorpusV2, style_vector
from forge.organism_raster_vae_v2.model import HierarchicalOrganismRasterVAE, hierarchical_loss
from forge.organism_raster_vae_v2.smoke import validate_smoke


@pytest.fixture(scope="module")
def corpus() -> OrganismRasterCorpusV2: return OrganismRasterCorpusV2()


def tiny_config() -> OrganismVAEV2Config:
    return OrganismVAEV2Config(width=32, coarse_width=64, coarse_latent_channels=8, fine_latent_channels=4, residual_depth=1, condition_dim=64)


def test_v2_contract_is_additive_and_materially_higher_capacity() -> None:
    assert len(source_sha256()) == 64 and len(source_manifest()) == 6
    assert set(authority()) == {"v1_data_contract_sha256", "frozen_upstream"}
    model = HierarchicalOrganismRasterVAE()
    assert sum(parameter.numel() for parameter in model.parameters()) == 35_612_578
    with pytest.raises(ValueError, match="dimensions"):
        OrganismVAEV2Config(width=16)


def test_style_condition_preserves_identity_appearance(corpus: OrganismRasterCorpusV2) -> None:
    styles = torch.stack([corpus[index]["style"] for index in range(len(corpus))])
    assert styles.shape == (45, 8) and torch.isfinite(styles).all() and bool(((styles >= 0) & (styles <= 1)).all())
    assert len(torch.unique(styles, dim=0)) == 45
    sample = corpus[0]
    assert torch.equal(style_vector(sample["rgba"], sample["occupancy"], sample["emission"]), sample["style"])


def test_hierarchical_latents_and_all_raster_heads_receive_gradients(corpus: OrganismRasterCorpusV2) -> None:
    config=tiny_config();model=HierarchicalOrganismRasterVAE(config);batch=default_collate([corpus[index] for index in range(2)]);generator=torch.Generator().manual_seed(17)
    output=model(batch["living_field"],batch["family"],batch["subtype"],batch["role"],batch["genes"],batch["style"],generator=generator)
    assert output.coarse_latent.shape==(2,8,12,12) and output.fine_latent.shape==(2,4,24,24)
    assert output.rgba.shape==(2,4,48,48) and output.coarse_occupancy_logits.shape==(2,1,24,24) and output.system_role_logits.shape==(2,8,4,48,48)
    loss,pieces=hierarchical_loss(output,batch,config);loss.backward()
    assert torch.isfinite(loss) and set(pieces)=={"loss","reconstruction","occupancy_bce","silhouette_dice_loss","rgb_l1","categorical_ce","physiology_l1","system_role_ce","cell_state_l1","emission_l1","coarse_occupancy_bce","palette_l1","edge_l1","symmetry_l1","kl_coarse","kl_fine"}
    for parameter in (model.coarse_mean.weight,model.fine_mean.weight,model.rgb.weight,model.physiology.weight,model.system_role.weight,model.cell_state.weight): assert parameter.grad is not None and float(parameter.grad.abs().sum())>0


def test_coarse_and_fine_edits_have_distinct_continuous_effects(corpus: OrganismRasterCorpusV2) -> None:
    config=tiny_config();model=HierarchicalOrganismRasterVAE(config).eval();batch=default_collate([corpus[0]])
    with torch.inference_mode():
        condition=model.condition_vector(batch["family"],batch["subtype"],batch["role"],batch["genes"],batch["style"]);cm,_,fm,_=model.encode(batch["living_field"],condition);base=model.decode(cm,fm,condition).rgba;coarse=model.decode(cm+.3,fm,condition).rgba;fine=model.decode(cm,fm+.3,condition).rgba
    assert not torch.equal(base,coarse) and not torch.equal(base,fine) and not torch.equal(coarse,fine)
    assert bool(((coarse>0)&(coarse<1)).all()) and bool(((fine>0)&(fine<1)).all())


def test_frozen_v2_output_exact_replay_when_present() -> None:
    output=Path("outputs/organism_raster_vae_v2/fit_v2")
    if not output.exists(): pytest.skip("Frozen v2 representation fit is not present in this checkout")
    report=validate_smoke(output);assert report["status"]=="passed" and report["gates"]["production_promotion_allowed"] is False
