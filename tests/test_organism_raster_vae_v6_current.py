from __future__ import annotations

import torch

from forge.organism_raster_vae_v5_anatomical.dataset import AnatomicalGraphCorpus
from forge.organism_raster_vae_v6_current.contract import PARENT_CHECKPOINT, TrainingPlan
from forge.organism_raster_vae_v6_current.training import _new_model, source_sha256


def test_current_corpus_differs_from_preserved_parent() -> None:
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=True)
    assert parent["corpus_sha256"] != AnatomicalGraphCorpus().semantic_sha256


def test_parent_warm_start_is_exact_and_source_is_stable() -> None:
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=True)
    model, checkpoint_sha = _new_model(torch.device("cpu"))
    assert checkpoint_sha and source_sha256() == source_sha256()
    for name, value in model.state_dict().items():
        assert torch.equal(value, parent["ema_state"][name])


def test_segment_plan_is_bounded() -> None:
    assert TrainingPlan().segment_steps == 100
