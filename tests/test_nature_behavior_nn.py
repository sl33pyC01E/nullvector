from __future__ import annotations

import numpy as np
import torch

from forge.nature_behavior_nn import ModelConfig,NeuralNatureBehavior
from forge.nature_behavior_nn.features import extract_observation
from forge.nature_sim_v2 import NatureWorld


def test_behavior_observation_and_transformer_shapes() -> None:
    world=NatureWorld(seed=21,size=32,max_population=32);world.seed_founders(variants_per_family=1);entity=world.organisms[1]
    self_features,resource,neighbor,mask=extract_observation(world,entity)
    assert self_features.shape==(94,) and resource.shape==(10,4) and neighbor.shape==(12,14) and mask.any()
    model=NeuralNatureBehavior(ModelConfig(width=64,layers=2,heads=4,dropout=0));result=model(torch.from_numpy(self_features[None]),torch.from_numpy(resource[None]),torch.from_numpy(neighbor[None]),torch.from_numpy(mask[None]))
    assert result.intent_logits.shape==(1,12) and result.direction.shape==(1,2) and result.urgency.shape==(1,)
    assert model.parameter_count>100_000 and torch.isfinite(result.direction).all()


def test_world_accepts_batched_neural_behavior_authority() -> None:
    class Policy:
        def __init__(self):self.actions={}
        def prepare(self,world):self.actions={item.entity_id:("guard",np.asarray((.25,-.5))) for item in world.organisms.values()}
        def choose(self,entity):return self.actions.get(entity.entity_id)
    world=NatureWorld(seed=22,size=32,max_population=32,behavior_policy=Policy());world.seed_founders(variants_per_family=1);world.step(.1)
    assert {item.intent for item in world.organisms.values()}=={"guard"}
    assert any(np.linalg.norm(item.velocity)>0 for item in world.organisms.values())
