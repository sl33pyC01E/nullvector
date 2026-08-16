from __future__ import annotations

import torch
import numpy as np

from forge.creature_stage_neural_grasper_v1.feeding import FeedingState, FoodClump
from forge.living_body_dynamics_nn import (
    BodyTransitionCorpus, LivingBodyDynamicsNet, NeuralLivingBodyDynamicsRuntime, collate_graphs,
)
from forge.living_body_dynamics_nn.contract import FEATURES, FEEDING_TARGETS
from forge.living_body_dynamics_nn.model import loss
from forge.living_body_substrate import LivingBody


def test_transition_corpus_covers_families_actions_and_causal_targets() -> None:
    corpus = BodyTransitionCorpus(repeats=1)
    rows = [corpus[identity * 35 + action] for identity in (0, 6, 12, 18, 24) for action in (0, 4)]
    assert {int(row["family"]) for row in rows} == set(range(5))
    for row in rows:
        assert row["features"].shape[1] == FEATURES
        assert row["edges"].shape[0] == 2
        assert row["target"].shape == (len(row["features"]), 3)
        assert row["feeding_target"].shape == (FEEDING_TARGETS,)
        assert torch.isfinite(row["features"]).all()
        assert torch.isfinite(row["target"]).all()


def test_graph_model_predicts_cells_and_whole_body_systems() -> None:
    corpus = BodyTransitionCorpus(repeats=1)
    batch = collate_graphs([corpus[index] for index in (0, 1, 48, 49)])
    model = LivingBodyDynamicsNet()
    cell, systems, feeding = model(batch)
    value, metrics = loss(model, batch)
    assert cell.shape == batch["target"].shape
    assert systems.shape == batch["systems"].shape
    assert feeding.shape == batch["feeding_target"].shape
    assert torch.isfinite(value)
    assert metrics["health_mae"] >= 0
    assert metrics["untouched_drift"] >= 0


def test_v3_feeding_head_hard_gates_and_conserves_matter() -> None:
    corpus = BodyTransitionCorpus(repeats=4)
    # identity 0, healthy scenario, feed action, repeats encode respectively:
    # valid contact, missed contact, incompatible diet, and full reserve.
    base = (0 * 5 * 7 * 4) + (0 * 7 * 4) + (4 * 4)
    batch = collate_graphs([corpus[base + repeat] for repeat in range(4)])
    model = LivingBodyDynamicsNet().eval()
    with torch.no_grad():
        _, _, feeding = model(batch)
    assert feeding[0, 0] > 0
    assert torch.equal(feeding[1:, 0], torch.zeros(3))
    for graph in range(4):
        node = int(torch.nonzero(batch["graph_index"] == graph, as_tuple=False)[0, 0])
        before_reserve = batch["features"][node, 50] * 4.0
        conversion = batch["features"][node, 48] * 6.0 * batch["features"][node, 49]
        expected_reserve = torch.clamp(before_reserve + feeding[graph, 0] * conversion, 0, 4.0) / 4.0
        assert torch.allclose(feeding[graph, 2], expected_reserve, atol=1e-6)
        mass = batch["features"][node, 47] * 3.0
        expected_mass = torch.clamp((mass - feeding[graph, 0]) / mass.clamp_min(1e-8), 0, 1)
        assert torch.allclose(feeding[graph, 8], expected_mass, atol=1e-6)


def test_corpus_replay_is_exact() -> None:
    left = BodyTransitionCorpus(repeats=1)[77]
    right = BodyTransitionCorpus(repeats=1)[77]
    for key in left:
        assert torch.equal(left[key], right[key])


def test_feeding_teacher_requires_contact_route_compatibility_and_capacity() -> None:
    corpus = BodyTransitionCorpus(repeats=4)

    def row(scenario: int, repeat: int):
        # identity=0, action=feed; repeats are the innermost corpus axis.
        return corpus[((scenario * 7 + 4) * 4) + repeat]["feeding_target"]

    healthy = row(0, 0)
    missed = row(0, 1)
    incompatible = row(0, 2)
    full = row(0, 3)
    feeder_ablation = row(2, 0)
    digestive_ablation = row(3, 0)
    assert healthy[6] == 1 and healthy[7] == 1 and healthy[0] > 0
    assert missed[6] == 0 and missed[0] == 0
    assert incompatible[6] == 1 and incompatible[0] == 0
    assert full[6] == 1 and full[0] == 0
    assert feeder_ablation[7] == 0 and feeder_ablation[0] == 0
    assert digestive_ablation[7] == 0 and digestive_ablation[0] == 0


def test_runtime_preserves_physical_contact_and_mass_constraints() -> None:
    corpus = BodyTransitionCorpus(repeats=1)
    body = LivingBody(corpus.organisms[0])
    feeding = FeedingState(reserve=.5, fullness_seconds=10)
    clump = FoodClump(np.zeros(2), np.zeros(2), 1.25, .4, 1.8, (1, 1, 1, 1, 1))
    runtime = NeuralLivingBodyDynamicsRuntime(LivingBodyDynamicsNet(), torch.device("cpu"))
    transition = runtime.predict(body, feeding, clump, contact=False, action_kind=4)
    assert transition.absorbed_mass == 0
    assert transition.nutrition == 0
    assert transition.clump_mass == clump.mass
    assert transition.health.shape == body.health.shape
    assert transition.safety_projected

    feeding.reserve = feeding.reserve_capacity
    transition = runtime.predict(body, feeding, clump, contact=True, action_kind=4)
    assert transition.absorbed_mass == 0 and transition.clump_mass == clump.mass
    feeding.reserve = .5
    clump.nutrition_by_family = (0, 1, 1, 1, 1)
    transition = runtime.predict(body, feeding, clump, contact=True, action_kind=4)
    assert transition.absorbed_mass == 0 and transition.nutrition == 0
