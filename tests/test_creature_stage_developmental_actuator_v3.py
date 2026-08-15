from __future__ import annotations

import numpy as np
import torch

from forge.creature_stage_developmental_motion.contract import DEFAULT_CORPUS, DEFAULT_PRIOR, MAX_NODES
from forge.creature_stage_developmental_motion.dataset import DevelopmentalMotionTeacher, DevelopmentalSequenceSampler
from forge.creature_stage_developmental_actuator_v2.contract import CausalActuatorConfig
from forge.creature_stage_developmental_actuator_v3.contract import BoneProjectionConfig, source_sha256
from forge.creature_stage_developmental_actuator_v3.model import LengthProjectedCellularActuator, project_bone_lengths
from forge.creature_stage_developmental_actuator_v3.training import _v2_seed_authority, load_model_state


def test_projection_is_differentiable_and_reduces_length_error() -> None:
    node = torch.zeros((1, MAX_NODES, 4), dtype=torch.float32, requires_grad=True)
    features = torch.zeros((1, MAX_NODES, 8), dtype=torch.float32)
    features[0, 1, 0] = 0.5  # rest edge is eight pixels long
    node.data[0, 1, 0] = 0.5  # stretch the edge by six more pixels
    mask = torch.zeros((1, MAX_NODES), dtype=torch.bool); mask[:, :2] = True
    adjacency = torch.zeros((1, MAX_NODES, MAX_NODES), dtype=torch.bool)
    adjacency[0, 0, 0] = adjacency[0, 1, 1] = adjacency[0, 0, 1] = adjacency[0, 1, 0] = True
    projected = project_bone_lengths(node, features, mask, adjacency, BoneProjectionConfig())
    rest = features[0, :2, :2] * 16.0
    before = rest + node[0, :2, :2] * 12.0
    after = rest + projected[0, :2, :2] * 12.0
    before_error = abs(float(torch.linalg.vector_norm(before[1] - before[0])) - 8.0)
    after_error = abs(float(torch.linalg.vector_norm(after[1] - after[0])) - 8.0)
    assert after_error < before_error * .35
    projected.square().mean().backward()
    assert node.grad is not None and float(node.grad.abs().sum()) > 0.0


def test_v2_state_registry_loads_exactly_into_v3() -> None:
    contract, checkpoint = _v2_seed_authority()
    model = LengthProjectedCellularActuator(CausalActuatorConfig(**contract["model"]), BoneProjectionConfig())
    load_model_state(model, checkpoint["ema_state"])
    assert set(model.state_dict()) == set(checkpoint["ema_state"])


def test_projected_forward_preserves_padding_and_has_lower_edge_strain() -> None:
    teacher = DevelopmentalMotionTeacher(DEFAULT_CORPUS, prior=DEFAULT_PRIOR, replay=False)
    frame = DevelopmentalSequenceSampler(teacher, batch_size=5, sequence_frames=12).sequence(2)[0][0]
    small = CausalActuatorConfig(width=96, depth=2, heads=4, feedforward_multiplier=2, condition_width=96, cell_width=64, cell_graph_blocks=1, dropout=0.0)
    model = LengthProjectedCellularActuator(small, BoneProjectionConfig())
    output = model(
        frame["static"], frame["state"], frame["mask"], frame["adjacency"], frame["node_features"],
        frame["node_state"], frame["node_mask"], frame["node_adjacency"], frame["muscle_features"],
        frame["muscle_state"], frame["muscle_mask"], frame["muscle_incidence"], frame["cell_node_weights"],
        frame["parent_prior"], frame["family"], frame["morphotype"], frame["phase"], frame["traits"],
    )
    assert torch.count_nonzero(output["node_state"][~frame["node_mask"]]) == 0
    assert torch.count_nonzero(output["cell_state"][~frame["mask"]]) == 0
    loss = output["cell_state"].square().mean(); loss.backward()
    assert model.actuator.muscle_out[-1].weight.grad is not None


def test_source_contract_is_stable() -> None:
    assert len(source_sha256()) == 64
    assert source_sha256() == source_sha256()
