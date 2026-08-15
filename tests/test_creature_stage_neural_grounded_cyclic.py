from __future__ import annotations

import math

import numpy as np
import torch

from forge.creature_stage_neural_grounded_cyclic.contract import (
    EVALUATION_IDENTITIES,
    TRAIN_IDENTITIES,
    CyclicModelConfig,
)
from forge.creature_stage_neural_grounded_cyclic.curriculum import CurriculumGroundedTeacher
from forge.creature_stage_neural_grounded_cyclic.dataset import RuntimeHonestGroundedTeacher
from forge.creature_stage_neural_grounded_cyclic.model import NeuralCyclicGroundedMotion, cyclic_phase_features
from forge.creature_stage_neural_grounded_cyclic.training import _load_parent, cyclic_loss
from forge.creature_stage_neural_motion.contract import CellularMotionTransformerConfig


def test_split_is_family_balanced_disjoint_and_graft_held_out() -> None:
    teacher = RuntimeHonestGroundedTeacher()
    assert teacher.split_indices("train") == TRAIN_IDENTITIES
    assert teacher.split_indices("validation") == EVALUATION_IDENTITIES
    assert set(TRAIN_IDENTITIES).isdisjoint(EVALUATION_IDENTITIES)
    assert teacher.arrays["grafted"][list(TRAIN_IDENTITIES)].tolist() == [0] * 5
    assert teacher.arrays["grafted"][list(EVALUATION_IDENTITIES)].tolist() == [1] * 5


def test_contact_offset_uses_current_not_target_cell() -> None:
    teacher = RuntimeHonestGroundedTeacher()
    found = False
    for identity in TRAIN_IDENTITIES:
        for frame in range(72):
            row = teacher.sample(identity, frame)
            active_cells = np.flatnonzero(row["dynamic"][:, 5] > .5)
            if not len(active_cells):
                continue
            cell = int(active_cells[0])
            owner = int(teacher.arrays["appendage_owner"][identity, cell])
            current = teacher.arrays["cells_local"][identity, (frame - 1) % 72, cell]
            expected = np.clip((teacher.arrays["contact_anchor_local"][identity, frame, owner] - current) / 24, -1, 1)
            assert np.allclose(row["dynamic"][cell, 6:8], expected)
            expected_body_x = np.clip((teacher.arrays["body_world_x"][identity, (frame - 1) % 72] - teacher.arrays["body_world_x"][identity, 0]) / 16, -1, 1)
            expected_clearance = np.clip((teacher.arrays["ground_y"][identity] - current[1]) / 32, -1, 1)
            assert np.isclose(row["dynamic"][cell, 11], expected_body_x)
            assert np.isclose(row["dynamic"][cell, 12], expected_clearance)
            found = True
            break
        if found:
            break
    assert found


def test_grafts_do_not_receive_an_untrained_morphotype_token() -> None:
    teacher = RuntimeHonestGroundedTeacher()
    for identity in range(10):
        row = teacher.sample(identity, 0)
        assert row["morphotype"] == row["family"] * 4


def test_small_physics_curriculum_is_balanced_and_component_conditioned() -> None:
    teacher = CurriculumGroundedTeacher(root=None, variants_per_family=2)
    assert [len(rows) for rows in teacher.family_indices] == [2] * 5
    assert teacher.arrays["cells_local"].shape == (10, 72, 560, 2)
    modes = set(teacher.arrays["locomotor_mode"][teacher.arrays["appendage_mask"].astype(bool)].tolist())
    assert modes == {0, 1, 2, 4}
    batch = teacher.batch(7, 5, torch.device("cpu"), split="train")
    assert batch["family"].tolist() == [0, 1, 2, 3, 4]


def test_cyclic_phase_features_are_exactly_periodic() -> None:
    values = cyclic_phase_features(torch.tensor([0.0, 1.0]), 8)
    assert torch.allclose(values[0], values[1], atol=1e-6, rtol=0)
    assert values.shape == (2, 16)


def test_cyclic_model_loads_exact_parent_and_backpropagates() -> None:
    teacher = RuntimeHonestGroundedTeacher()
    config = CellularMotionTransformerConfig(width=64, depth=2, heads=4, condition_width=128, dropout=0)
    compact = NeuralCyclicGroundedMotion(config, CyclicModelConfig(refinement_width=128, refinement_depth=2))
    batch = teacher.batch(2, 5, torch.device("cpu"), split="train")
    result = compact(
        batch["static"], batch["state"], batch["dynamic"], batch["mask"], batch["adjacency"],
        batch["family"], batch["morphotype"], batch["motion"], batch["phase"], batch["controls"],
    )
    loss, pieces = cyclic_loss(result, batch, batch["state"])
    loss.backward()
    assert result.cells.shape == (5, 560, 4)
    assert float(result.direct_gate[batch["mask"]].min()) >= .84
    assert math.isfinite(float(loss)) and float(pieces["outside"]) == 0
    assert any(parameter.grad is not None for parameter in compact.parameters())


def test_production_shape_model_accepts_sealed_rollout_parent() -> None:
    model = NeuralCyclicGroundedMotion(CellularMotionTransformerConfig(), CyclicModelConfig())
    parent = _load_parent(model)
    assert parent["update"] == 1000
    assert parent["sha256"] == "157a29eaee49221523ecc97dd7ba758461d3472930f9321f3918b1b4dd352513"
