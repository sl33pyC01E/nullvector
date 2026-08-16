from __future__ import annotations

import torch

from forge.creature_stage_neural_locomotion_25d.contract import DYNAMIC_FEATURES,MAX_APPENDAGES,MAX_MUSCLES,ModelConfig
from forge.creature_stage_neural_locomotion_25d.model import NeuralLocomotion25D
from forge.creature_stage_neural_locomotion_25d.runtime import NeuralLocomotionRuntime
from forge.nature_sim_v2 import NatureWorld,founder_genomes


def test_sequence_controller_shapes_and_gradients() -> None:
    model=NeuralLocomotion25D(ModelConfig(width=192,recurrent_layers=2,dropout=0))
    batch,time=2,8
    output=model(torch.zeros(batch,20),torch.zeros(batch,MAX_APPENDAGES,16),torch.ones(batch,MAX_APPENDAGES,dtype=torch.bool),torch.zeros(batch,MAX_MUSCLES,8),torch.zeros(batch,MAX_MUSCLES,dtype=torch.long),torch.ones(batch,MAX_MUSCLES,dtype=torch.bool),torch.zeros(batch,time,DYNAMIC_FEATURES))
    assert output.contact_logits.shape==(batch,time,MAX_APPENDAGES)
    assert output.muscle.shape==(batch,time,MAX_MUSCLES)
    assert output.velocity.shape==(batch,time,2)
    (output.muscle.mean()+output.contact_logits.mean()+output.velocity.mean()).backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_online_runtime_populates_physical_control_channels() -> None:
    model=NeuralLocomotion25D(ModelConfig(width=192,recurrent_layers=2,dropout=0))
    runtime=NeuralLocomotionRuntime(model,device="cpu")
    world=NatureWorld(seed=5,size=32,max_population=30,motion_policy=runtime)
    entity_id=world.add_organism(founder_genomes(variants_per_family=1)[1],(10,10),energy=.8)
    world.step(.1)
    entity=world.organisms[entity_id]
    assert entity.neural_contacts.shape==(len(entity.body.organism.genome.appendages),)
    assert entity.neural_muscles.shape==(len(entity.body.organism.muscles),)
    assert entity.finite()
