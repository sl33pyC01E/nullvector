from __future__ import annotations

import torch

from forge.recurrent_world_student_v5.contract import CORPUS, PARENT_SHA256, TrainingPlan
from forge.recurrent_world_student_v5.model import PerceptionRecurrentWorldStudent
from forge.world_latent_dit.contract import ModelConfig


def test_v5_binds_natural_corpus_and_frozen_parent() -> None:
    assert "world_action_natural_v10" in CORPUS.as_posix()
    assert len(PARENT_SHA256) == 64
    assert TrainingPlan().batch_size == 4


def test_perception_adapter_preserves_parent_at_initialization() -> None:
    torch.manual_seed(1)
    model = PerceptionRecurrentWorldStudent(ModelConfig(width=32, layers=1, heads=4, patch=4)).eval()
    current = torch.randn(2,48,32,32); previous=torch.randn_like(current); action=torch.zeros(2,dtype=torch.long); control=torch.zeros(2,4); state=torch.zeros(2,64); actor=torch.zeros(2,128); visibility=torch.rand(2,1,32,32); memory=torch.rand(2,1,32,32)
    with torch.inference_mode():
        expected=model.world.action(current,previous,action,control,state,actor)
        actual=model.action(current,previous,action,control,state,actor,visibility,memory)
    torch.testing.assert_close(actual,expected)


def test_perception_adapter_has_trainable_spatial_and_summary_paths() -> None:
    model=PerceptionRecurrentWorldStudent(ModelConfig(width=32,layers=1,heads=4,patch=4))
    names={name for name,_ in model.named_parameters()}
    assert "perception.spatial.2.weight" in names
    assert "perception.summary.2.weight" in names
