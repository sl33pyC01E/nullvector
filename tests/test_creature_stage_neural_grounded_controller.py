from __future__ import annotations

import numpy as np
import pytest
import torch

from forge.creature_stage_neural_grounded_components.dataset import ComponentSentinelTeacher
from forge.creature_stage_neural_grounded_controller.contract import MAX_APPENDAGES, MAX_MUSCLES, ControllerModelConfig, source_sha256
from forge.creature_stage_neural_grounded_controller.dataset import MUSCLE_META_FEATURES, OWNER_META_FEATURES, POOLED_FEATURES, muscle_metadata, owner_metadata
from forge.creature_stage_neural_grounded_controller.model import NeuralGroundedController
from forge.creature_stage_neural_grounded_controller.physics import simulate_controlled_cycle
from forge.creature_stage_neural_grounded_controller.evaluation import validate_evaluation


def test_metadata_and_model_contract() -> None:
    teacher = ComponentSentinelTeacher(); organism = teacher.organisms[1]
    owner_meta, owner_mask = owner_metadata(organism); muscle_meta, muscle_owner, muscle_mask = muscle_metadata(organism)
    assert owner_meta.shape == (MAX_APPENDAGES, OWNER_META_FEATURES)
    assert muscle_meta.shape == (MAX_MUSCLES, MUSCLE_META_FEATURES)
    model = NeuralGroundedController(ControllerModelConfig(width=128, depth=3, dropout=0))
    output = model(
        torch.zeros(1, MAX_APPENDAGES, POOLED_FEATURES), torch.zeros(1, POOLED_FEATURES),
        torch.from_numpy(owner_meta)[None], torch.from_numpy(owner_mask)[None],
        torch.from_numpy(muscle_meta)[None], torch.from_numpy(muscle_owner)[None], torch.from_numpy(muscle_mask)[None],
    )
    assert output.muscle_activation.shape == (1, MAX_MUSCLES)
    assert output.contact_logits.shape == (1, MAX_APPENDAGES)
    assert output.body_velocity.shape == (1,)
    assert torch.count_nonzero(output.muscle_activation[0, ~torch.from_numpy(muscle_mask)]) == 0
    with pytest.raises(ValueError, match="input drifted"):
        model(torch.zeros(1, 1, 1), torch.zeros(1, POOLED_FEATURES), torch.from_numpy(owner_meta)[None], torch.from_numpy(owner_mask)[None], torch.from_numpy(muscle_meta)[None], torch.from_numpy(muscle_owner)[None], torch.from_numpy(muscle_mask)[None])


def test_controlled_physics_replays_teacher_controls() -> None:
    teacher = ComponentSentinelTeacher(); identity = 1; organism = teacher.organisms[identity]
    appendages = len(organism.genome.appendages); muscles = len(organism.muscles)
    cycle = simulate_controlled_cycle(
        organism,
        teacher.arrays["contact_active"][identity, :, :appendages].astype(np.bool_),
        teacher.arrays["muscle_activation"][identity, :, :muscles].astype(np.float32),
    )
    count = organism.cell_count
    expected = teacher.arrays["cells_local"][identity, :, :count]
    actual = np.stack([frame.cells_local for frame in cycle.frames])
    assert np.array_equal(actual, expected)
    assert cycle.maximum_contact_slip_px < .05
    assert cycle.maximum_edge_strain < .12
    assert cycle.loop_seam_max_abs < .002
    assert cycle.vertical_axis_max_degrees < 5


def test_controller_source_is_bound() -> None:
    assert len(source_sha256()) == 64


def test_reviewed_evaluation_contract_when_present() -> None:
    from forge.config import PROJECT_ROOT
    output = PROJECT_ROOT / "outputs/creature_stage_neural_grounded_controller/evaluation_0800_final_verified"
    if not output.exists():
        pytest.skip("reviewed controller evaluation is not checked out")
    report = validate_evaluation(output, replay=False)
    assert report["status"] == "passed"
    assert report["promotion_eligible"] is True
    assert all(report["gates"].values())
