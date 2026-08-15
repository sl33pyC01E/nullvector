from __future__ import annotations

import torch

from forge.creature_stage_neural_grounded_components.contract import MAX_APPENDAGES, PARENT, PARENT_SHA256, ComponentModelConfig
from forge.creature_stage_neural_grounded_components.dataset import ComponentCurriculumTeacher, ComponentSentinelTeacher
from forge.creature_stage_neural_grounded_components.model import NeuralComponentGroundedMotion
from forge.creature_stage_neural_grounded_components.training import _forward, component_loss
from forge.creature_stage_neural_grounded_cyclic.contract import CyclicModelConfig, sha256_file
from forge.creature_stage_neural_motion.contract import CellularMotionTransformerConfig


def test_component_batches_preserve_exact_appendage_owners() -> None:
    teacher = ComponentCurriculumTeacher(); batch = teacher.batch(4, 5, torch.device("cpu"))
    assert batch["owner"].shape == (5, 560) and batch["owner"].dtype == torch.long
    assert int(batch["owner"].min()) == -1 and int(batch["owner"].max()) < MAX_APPENDAGES
    for row, identity in enumerate(batch["identity"].tolist()):
        expected = torch.from_numpy(teacher.arrays["appendage_owner"][identity].astype("int64"))
        assert torch.equal(batch["owner"][row], expected)


def test_component_model_pools_and_backpropagates() -> None:
    teacher=ComponentCurriculumTeacher();batch=teacher.batch(8,5,torch.device("cpu"))
    model=NeuralComponentGroundedMotion(CellularMotionTransformerConfig(width=64,depth=2,heads=4,condition_width=128,dropout=0),CyclicModelConfig(refinement_width=128,refinement_depth=2),ComponentModelConfig(width=128,depth=2))
    result=_forward(model,batch,batch["state"]);loss,pieces=component_loss(result,batch,batch["state"]);loss.backward()
    assert result.cells.shape==(5,560,4) and result.owner_translation.shape==(5,8,2)
    assert float(pieces["outside"])==0 and any(parameter.grad is not None for parameter in model.parameters())


def test_sentinel_owners_cover_original_grafts_and_parent_is_exact() -> None:
    teacher=ComponentSentinelTeacher();device=torch.device("cpu")
    for identity in teacher.split_indices("validation"):
        owner=teacher.owner(identity,device);assert owner.shape==(1,560);assert int(owner.max())<MAX_APPENDAGES
    assert sha256_file(PARENT)==PARENT_SHA256
