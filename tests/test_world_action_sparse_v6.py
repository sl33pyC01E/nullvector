from __future__ import annotations

import torch

from forge.world_action_sparse_v6.training import _selection


def test_validation_selection_prefers_persistence_beating_causal_editor():
    good = {"model_latent_mae": .08, "persistence_latent_mae": .10, "changed_model_latent_mae": .20, "changed_persistence_latent_mae": .25, "correct_action_advantage": .03, "targeted_control_advantage": .01}
    bad = {**good, "model_latent_mae": .12, "changed_model_latent_mae": .29}
    assert _selection(good) < _selection(bad)


def test_validation_selection_penalizes_inverted_counterfactuals():
    causal = {"model_latent_mae": .09, "persistence_latent_mae": .10, "changed_model_latent_mae": .22, "changed_persistence_latent_mae": .25, "correct_action_advantage": .02, "targeted_control_advantage": .01}
    inverted = {**causal, "correct_action_advantage": -.03, "targeted_control_advantage": -.02}
    assert _selection(causal) < _selection(inverted)
