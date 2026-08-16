from __future__ import annotations

import torch

from forge.creature_stage_neural_locomotion_25d.contract import DYNAMIC_FEATURES,MAX_APPENDAGES,MAX_MUSCLES,ModelConfig
from forge.creature_stage_neural_locomotion_25d.model import NeuralLocomotion25D


def test_sequence_controller_shapes_and_gradients() -> None:
    model=NeuralLocomotion25D(ModelConfig(width=192,recurrent_layers=2,dropout=0))
    batch,time=2,8
    output=model(torch.zeros(batch,20),torch.zeros(batch,MAX_APPENDAGES,16),torch.ones(batch,MAX_APPENDAGES,dtype=torch.bool),torch.zeros(batch,MAX_MUSCLES,8),torch.zeros(batch,MAX_MUSCLES,dtype=torch.long),torch.ones(batch,MAX_MUSCLES,dtype=torch.bool),torch.zeros(batch,time,DYNAMIC_FEATURES))
    assert output.contact_logits.shape==(batch,time,MAX_APPENDAGES)
    assert output.muscle.shape==(batch,time,MAX_MUSCLES)
    assert output.velocity.shape==(batch,time,2)
    (output.muscle.mean()+output.contact_logits.mean()+output.velocity.mean()).backward()
    assert all(parameter.grad is not None for parameter in model.parameters())

