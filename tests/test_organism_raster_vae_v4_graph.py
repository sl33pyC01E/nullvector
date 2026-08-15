from __future__ import annotations

import torch

from forge.organism_raster_vae_v3.contract import RasterVAEV3Config
from forge.organism_raster_vae_v4_graph.dataset import MAX_TOKENS,TOKEN_FEATURES,GraphTokenCorpus
from forge.organism_raster_vae_v4_graph.model import GraphTokenRasterVAE,loss


def test_graph_token_geometry_and_owner_alignment() -> None:
    row=GraphTokenCorpus()[96];assert row["tokens"].shape==(MAX_TOKENS,TOKEN_FEATURES);assert row["token_mask"].dtype==torch.bool;assert row["token_owner"].shape==(48,48);assert int(row["token_owner"].max())<int(row["token_mask"].sum())


def test_graph_model_forward_and_owner_loss() -> None:
    corpus=GraphTokenCorpus();rows=[corpus[i] for i in (96,97)];batch={key:torch.stack([row[key] for row in rows]) for key in rows[0]};config=RasterVAEV3Config();model=GraphTokenRasterVAE(config);output=model(batch["living"],batch["family"],batch["traits"],batch["phase"],batch["tokens"],batch["token_mask"],stochastic=False);value,metrics=loss(output,batch,config,.1);assert output.token_attention.shape==(2,24*24,MAX_TOKENS);assert torch.isfinite(value);assert metrics["token_owner_nll"]>0
