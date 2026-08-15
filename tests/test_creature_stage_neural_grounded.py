from __future__ import annotations

import numpy as np
import torch

from forge.creature_stage_neural_grounded.contract import MAX_CELLS, ROLLOUT_PARENT
from forge.creature_stage_neural_grounded.dataset import GroundedMotionTeacher
from forge.creature_stage_neural_grounded.model import NeuralGroundedMotion, grounded_loss
from forge.creature_stage_neural_grounded.raster import living_field_from_cells
from forge.creature_stage_neural_grounded.training import _load_parent
from forge.creature_stage_neural_motion.contract import CellularMotionTransformerConfig


def test_grounded_teacher_census_modes_and_splits() -> None:
    teacher = GroundedMotionTeacher()
    assert teacher.split_indices("train") == (0, 2, 4, 6, 8)
    assert teacher.split_indices("validation") == (1, 3, 5, 7, 9)
    assert teacher.arrays["cell_mask"].sum(1).tolist() == [267, 353, 295, 341, 492, 560, 377, 419, 404, 418]
    assert teacher.arrays["family"].tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    assert set(teacher.arrays["locomotor_mode"][teacher.arrays["appendage_mask"].astype(bool)].tolist()) == {0, 1, 2, 4}


def test_grounded_sample_is_loop_continuous_and_contact_conditioned() -> None:
    teacher = GroundedMotionTeacher()
    row0 = teacher.sample(0, 0); row71 = teacher.sample(0, 71)
    assert row0["static"].shape == (MAX_CELLS, 61)
    assert row0["dynamic"].shape == (MAX_CELLS, 16)
    assert np.array_equal(row0["state"][:, :2], row71["target"][:, :2])
    assert bool((row0["dynamic"][:, 5] > 0).any())


def test_expanded_backbone_loads_sealed_rollout_and_backpropagates() -> None:
    teacher = GroundedMotionTeacher(); device = torch.device("cpu")
    model = NeuralGroundedMotion(CellularMotionTransformerConfig(width=64, depth=2, heads=4, condition_width=128, dropout=0))
    # Shape behavior is tested with the compact model.  Production parent
    # loading is independently bound to the exact 27M-parameter authority.
    batch = teacher.batch(1, 5, device, split="train")
    result = model(batch["static"], batch["state"], batch["dynamic"], batch["mask"], batch["adjacency"], batch["family"], batch["morphotype"], batch["motion"], batch["phase"], batch["controls"])
    loss, pieces = grounded_loss(result, batch)
    loss.backward()
    assert result.cells.shape == (5, MAX_CELLS, 4)
    assert result.body_velocity.shape == (5,)
    assert float(pieces["outside"]) == 0
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_grounded_living_field_preserves_cell_authority() -> None:
    teacher = GroundedMotionTeacher(); row = teacher.sample(7, 12); count = row["cell_count"]
    condition = living_field_from_cells(
        teacher.arrays["cells_local"][7, 12, :count], teacher.arrays["tissue"][7, :count],
        teacher.arrays["trait_fields"][7, :count], teacher.arrays["appendage_owner"][7, :count], 3,
    )
    assert condition.living.shape == (1, 74, 48, 48)
    assert 0 < int((condition.living[0, 0] > 0).sum()) <= count
    assert torch.isfinite(condition.living).all()
    assert torch.equal(condition.living[0, 0], condition.living[0, 73])
