from __future__ import annotations

import torch

from forge.living_body_dynamics_nn import BodyTransitionCorpus, LivingBodyDynamicsNet, collate_graphs
from forge.living_body_dynamics_nn.contract import FEATURES
from forge.living_body_dynamics_nn.model import loss


def test_transition_corpus_covers_families_actions_and_causal_targets() -> None:
    corpus = BodyTransitionCorpus(repeats=1)
    rows = [corpus[index] for index in (0, 3, 72, 75, 144, 147, 216, 219, 288, 291)]
    assert {int(row["family"]) for row in rows} == set(range(5))
    for row in rows:
        assert row["features"].shape[1] == FEATURES
        assert row["edges"].shape[0] == 2
        assert row["target"].shape == (len(row["features"]), 3)
        assert torch.isfinite(row["features"]).all()
        assert torch.isfinite(row["target"]).all()


def test_graph_model_predicts_cells_and_whole_body_systems() -> None:
    corpus = BodyTransitionCorpus(repeats=1)
    batch = collate_graphs([corpus[index] for index in (0, 1, 48, 49)])
    model = LivingBodyDynamicsNet()
    cell, systems = model(batch)
    value, metrics = loss(model, batch)
    assert cell.shape == batch["target"].shape
    assert systems.shape == batch["systems"].shape
    assert torch.isfinite(value)
    assert metrics["health_mae"] >= 0
    assert metrics["untouched_drift"] >= 0


def test_corpus_replay_is_exact() -> None:
    left = BodyTransitionCorpus(repeats=1)[77]
    right = BodyTransitionCorpus(repeats=1)[77]
    for key in left:
        assert torch.equal(left[key], right[key])
