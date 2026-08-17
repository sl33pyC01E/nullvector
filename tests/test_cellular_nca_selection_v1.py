from __future__ import annotations

import pytest
import torch

from forge.cellular_nca_selection_v1.evaluation import _gates, _state_hash


def test_candidate_gates_require_all_organs_and_rollout_quality() -> None:
    metrics = {
        "counterfactual_mae": .01,
        "organ_counterfactuals": [{"relative_change": value} for value in (-.1, -.01, -.02, -.4)],
        "rollout_mae": {"health": .01, "fluid": .02, "neural_activity": .03},
    }
    assert all(_gates(metrics, .05).values())
    metrics["organ_counterfactuals"][1]["relative_change"] = 0
    assert not _gates(metrics, .05)["all_four_organs_reduce_readout"]


def test_state_hash_is_sensitive() -> None:
    first = {"weight": torch.zeros(2)}; second = {"weight": torch.tensor([0.0, 1.0])}
    assert _state_hash(first) != _state_hash(second)
