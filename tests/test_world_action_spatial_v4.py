from __future__ import annotations

import numpy as np
import torch

from forge.world_action_spatial_v4 import ModelConfig, SpatialActionDiT, SpatialWorldActionRuntime, spatial_control_fields
from forge.world_action_spatial_v4.data import causal_action_and_settle_mask


def test_spatial_fields_place_actor_center_and_aim_in_command_direction():
    control=torch.tensor([[0.0,1.0,0.75,-0.5]],dtype=torch.float32)
    fields=spatial_control_fields(control,32)
    assert fields.shape==(1,4,32,32)
    actor_index=int(fields[0,0].argmax());aim_index=int(fields[0,1].argmax())
    actor_y,actor_x=divmod(actor_index,32);aim_y,aim_x=divmod(aim_index,32)
    assert abs(actor_x-16)<=1 and abs(actor_y-16)<=1
    assert aim_x>actor_x and aim_y<actor_y
    assert float(fields[0,2,actor_y,actor_x])>.85


def test_spatial_action_runtime_preserves_latent_shape():
    model=SpatialActionDiT(ModelConfig(width=64,layers=1,heads=4,patch=4)).eval()
    runtime=SpatialWorldActionRuntime(model,torch.device("cpu"),{},torch.zeros(1,48,1,1),torch.ones(1,48,1,1))
    current=torch.randn(2,48,32,32)
    result=runtime.predict_latent(current,action=np.asarray([2,6]),control=np.asarray([[0,0,.4,-.2],[1,0,-.7,.5]],np.float32),state=np.zeros((2,64),np.float32))
    assert result.shape==current.shape
    assert torch.isfinite(result).all()
    assert torch.equal(result,current)


def test_spatial_corpus_excludes_setup_jumps_but_keeps_actions_and_settle():
    actions=np.asarray([2,0,0,6,0,0,5,0],np.uint8)
    keep=causal_action_and_settle_mask(actions)
    assert np.array_equal(keep,np.asarray([1,1,0,1,1,0,1,1],bool))
