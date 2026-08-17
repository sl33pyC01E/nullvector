from __future__ import annotations

import torch

from forge.world_rssm_v1 import ModelConfig, RecurrentWorldStudent, TrainingPlan


def test_recurrent_world_student_shapes_and_gradients():
    model = RecurrentWorldStudent(ModelConfig(hidden=32, condition=24))
    current = torch.randn(2, 48, 32, 32); previous = torch.randn_like(current)
    result = model(current, previous, torch.tensor([0, 1]), torch.zeros(2, 4), torch.zeros(2, 64), torch.zeros(2, 128))
    assert result.latent.shape == current.shape
    assert result.actor_state.shape == (2, 128)
    assert result.hidden.shape == (2, 32, 16, 16)
    result.latent.mean().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_training_plan_is_segmented():
    assert TrainingPlan(total_updates=500, segment_updates=250).total_updates == 500
