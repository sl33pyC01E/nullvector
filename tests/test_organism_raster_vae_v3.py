from __future__ import annotations

import torch

from forge.organism_raster_vae_v3.contract import INPUT_CHANNELS,RasterVAEV3Config
from forge.organism_raster_vae_v3.dataset import MorphologyMotionCorpus
from forge.organism_raster_vae_v3.model import StructuredRasterVAE,loss


def test_corpus_is_family_phase_balanced() -> None:
    corpus=MorphologyMotionCorpus()
    assert len(corpus)==30*16
    row=corpus[0]
    assert row["living"].shape==(INPUT_CHANNELS,48,48)
    assert row["rgba"].shape==(4,96,96)
    assert row["occupancy"].shape==(48,48)
    assert torch.isfinite(row["living"]).all() and torch.isfinite(row["rgba"]).all()
    assert 0<=float(row["rgba"].min())<=float(row["rgba"].max())<=1


def test_model_forward_loss_and_capacity() -> None:
    corpus=MorphologyMotionCorpus(); rows=[corpus[i] for i in (0,97)]
    batch={key:torch.stack([row[key] for row in rows]) for key in rows[0]}
    config=RasterVAEV3Config(); model=StructuredRasterVAE(config)
    output=model(batch["living"],batch["family"],batch["traits"],batch["phase"],stochastic=False)
    assert output.rgba.shape==(2,4,96,96)
    assert output.occupancy_logits.shape==(2,1,48,48)
    assert output.tissue_logits.shape==(2,15,48,48)
    value,metrics=loss(output,batch,config,.1)
    assert torch.isfinite(value) and all(torch.isfinite(torch.tensor(item)) for item in metrics.values())
    parameters=sum(parameter.numel() for parameter in model.parameters())
    assert parameters>50_000_000
