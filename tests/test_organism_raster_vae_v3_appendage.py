from __future__ import annotations

import torch

from forge.organism_raster_vae_v3.contract import RasterVAEV3Config
from forge.organism_raster_vae_v3_appendage.dataset import APPENDAGE_CLASSES,INPUT_CHANNELS,AppendageMotionCorpus
from forge.organism_raster_vae_v3_appendage.model import AppendageRasterVAE,loss


def test_appendage_corpus_alignment() -> None:
    corpus=AppendageMotionCorpus(); row=corpus[96]
    assert row["living"].shape==(INPUT_CHANNELS,48,48)
    assert row["appendage"].shape==(48,48) and int(row["appendage"].max())<len(APPENDAGE_CLASSES)
    assert row["appendage_alpha"].shape==(1,96,96) and float(row["appendage_alpha"].sum())>0


def test_appendage_model_loss() -> None:
    corpus=AppendageMotionCorpus(); rows=[corpus[i] for i in (96,97)]; batch={key:torch.stack([row[key] for row in rows]) for key in rows[0]}; config=RasterVAEV3Config(); model=AppendageRasterVAE(config); output=model(batch["living"],batch["family"],batch["traits"],batch["phase"],stochastic=False); value,metrics=loss(output,batch,config,.1)
    assert output.appendage_logits.shape==(2,len(APPENDAGE_CLASSES),48,48); assert output.appendage_alpha_logits.shape==(2,1,96,96); assert torch.isfinite(value); assert metrics["appendage_ce"]>0
