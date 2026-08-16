from __future__ import annotations

import torch
from pathlib import Path

from forge.nature_colony_nn import FEATURES,MAX_MEMBERS,ColonyCoordinator,ModelConfig,NeuralColonyRuntime
from forge.nature_colony_nn.corpus import build_corpus


def test_colony_corpus_is_deterministic_balanced_and_heldout_ready() -> None:
    left=build_corpus(colonies=80,seed=4);right=build_corpus(colonies=80,seed=4)
    assert left["semantic_sha256"]==right["semantic_sha256"]
    assert left["features"].shape==(80,MAX_MEMBERS,FEATURES)
    assert left["mask"].sum()>500 and set(left["roles"][left["mask"]].tolist())==set(range(6))


def test_colony_transformer_jointly_predicts_members() -> None:
    corpus=build_corpus(colonies=8,seed=9);model=ColonyCoordinator(ModelConfig(width=96,layers=2,heads=4));features=torch.from_numpy(corpus["features"]);mask=torch.from_numpy(corpus["mask"]);roles,actions=model(features,mask)
    assert roles.shape==(8,MAX_MEMBERS,6) and actions.shape==(8,MAX_MEMBERS,3)
    assert torch.isfinite(roles).all() and torch.all((actions>=0)&(actions<=1))


def test_published_colony_checkpoint_is_current_and_loadable() -> None:
    path=Path(__file__).resolve().parents[1]/"game/generated/models/nature_colony/coordinator_v1.pt"
    runtime=NeuralColonyRuntime.from_checkpoint(path,device="cpu")
    assert sum(parameter.numel() for parameter in runtime.model.parameters())==10_726_025
