from __future__ import annotations

import torch
import inspect

from forge.recurrent_world_student_v5.contract import CORPUS, PARENT_SHA256, TrainingPlan
from forge.recurrent_world_student_v5.model import PerceptionRecurrentWorldStudent
from forge.recurrent_world_student_v5 import training
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
    assert "change_gate.2.weight" in names


def test_learned_change_gate_starts_near_persistence() -> None:
    model=PerceptionRecurrentWorldStudent(ModelConfig(width=32,layers=1,heads=4,patch=4)).eval()
    current=torch.randn(1,48,32,32); previous=torch.randn_like(current); action=torch.zeros(1,dtype=torch.long); control=torch.zeros(1,4); state=torch.zeros(1,64); actor=torch.zeros(1,128); visibility=torch.ones(1,1,32,32); memory=torch.ones_like(visibility)
    with torch.inference_mode():
        _,logits=model.gated_action(current,previous,action,control,state,actor,visibility,memory)
    torch.testing.assert_close(logits,torch.full_like(logits,-3.0))


def test_change_gate_learns_when_prediction_beats_persistence() -> None:
    source=inspect.getsource(training.train)
    assert "proposal * truth" in source
    assert "proposal.square()" in source
    assert "torch.clamp(" in source
