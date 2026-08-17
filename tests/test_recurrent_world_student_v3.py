from __future__ import annotations

from forge.recurrent_world_student_v3.model import RecurrentWorldStudent
from forge.world_latent_dit.contract import ModelConfig


def test_recurrent_world_student_contains_action_and_actor_models():
    model = RecurrentWorldStudent(ModelConfig(width=64, layers=1, heads=4))
    assert model.action.parameter_count > 0
    assert model.actor.parameter_count > 0
    assert model.parameter_count == model.action.parameter_count + model.actor.parameter_count
