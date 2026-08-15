from __future__ import annotations

from pathlib import Path

import torch

from forge.creature_stage_developmental_motion.contract import DEFAULT_CORPUS, DEFAULT_PRIOR
from forge.creature_stage_developmental_motion.dataset import DevelopmentalMotionTeacher, DevelopmentalSequenceSampler
from forge.creature_stage_developmental_actuator_v2.contract import CausalActuatorConfig, CausalTrainingConfig, source_sha256
from forge.creature_stage_developmental_actuator_v2.model import MuscleCausalCellularActuator, causal_actuator_loss
from forge.creature_stage_developmental_actuator_v2.training import (
    _v1_seed_authority,
    make_model,
    prepare_production,
    warm_start_from_v1,
)


def _frame():
    teacher = DevelopmentalMotionTeacher(DEFAULT_CORPUS, prior=DEFAULT_PRIOR, replay=False)
    return DevelopmentalSequenceSampler(teacher, batch_size=5, sequence_frames=12).sequence(3)[0][0]


def _small_model() -> MuscleCausalCellularActuator:
    return MuscleCausalCellularActuator(CausalActuatorConfig(
        width=96, depth=2, heads=4, feedforward_multiplier=2,
        condition_width=96, cell_width=64, cell_graph_blocks=1, dropout=0.0,
    ))


def test_source_contract_is_stable() -> None:
    assert len(source_sha256()) == 64
    assert source_sha256() == source_sha256()


def test_v1_warm_start_is_bounded_and_exactly_scoped() -> None:
    _, checkpoint = _v1_seed_authority()
    model = make_model()
    result = warm_start_from_v1(model, checkpoint["ema_state"])
    assert result["transferred_tensors"] > 100
    assert len(result["new_tensors"]) == 6
    assert all(name.startswith("actuator.") for name in result["new_tensors"])


def test_predicted_muscle_has_same_frame_causal_gradient_to_joint_motion() -> None:
    frame = _frame()
    model = _small_model()
    output = model(
        frame["static"], frame["state"], frame["mask"], frame["adjacency"],
        frame["node_features"], frame["node_state"], frame["node_mask"], frame["node_adjacency"],
        frame["muscle_features"], frame["muscle_state"], frame["muscle_mask"],
        frame["muscle_incidence"], frame["cell_node_weights"], frame["parent_prior"],
        frame["family"], frame["morphotype"], frame["phase"], frame["traits"],
    )
    joint_objective = output["node_state"][:, :, :2].square().mean()
    joint_objective.backward()
    muscle_head_gradients = [
        parameter.grad for parameter in model.actuator.muscle_out.parameters()
        if parameter.grad is not None
    ]
    assert muscle_head_gradients
    assert sum(float(gradient.abs().sum()) for gradient in muscle_head_gradients) > 0.0
    assert float(torch.sigmoid(model.actuator.previous_muscle_gate)) < .11
    assert torch.count_nonzero(output["muscle_node_force"][frame["node_mask"]]) > 0


def test_causal_loss_is_finite_and_updates_new_force_path() -> None:
    frame = _frame()
    model = _small_model()
    output = model(
        frame["static"], frame["state"], frame["mask"], frame["adjacency"],
        frame["node_features"], frame["node_state"], frame["node_mask"], frame["node_adjacency"],
        frame["muscle_features"], frame["muscle_state"], frame["muscle_mask"],
        frame["muscle_incidence"], frame["cell_node_weights"], frame["parent_prior"],
        frame["family"], frame["morphotype"], frame["phase"], frame["traits"],
    )
    loss, pieces = causal_actuator_loss(
        output, frame, frame["state"], frame["node_state"], frame["muscle_state"],
        CausalTrainingConfig(sequence_frames=12),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert set(("muscle_l1", "muscle_velocity", "muscle_force")) <= set(pieces)
    assert model.actuator.force_to_node[0].weight.grad is not None
    assert float(model.actuator.force_to_node[0].weight.grad.abs().sum()) > 0.0


def test_prepare_production_is_canonical_and_idempotent(tmp_path: Path) -> None:
    output = tmp_path / "production"
    first = prepare_production(output, total_updates=500, segment_updates=50, batch_size=5)
    second = prepare_production(output, total_updates=500, segment_updates=50, batch_size=5)
    assert first == second
    assert first["v1_seed"]["transferred_tensors"] > 100
    assert first["training"]["teacher_forcing_end"] == 0.0
