from __future__ import annotations

import torch

from forge.map_topology_neural_prior_v3 import frequency_weights, semantic_topology_loss


def _case(good: bool):
    logits=torch.full((1,3,5,5),-5.0);logits[:,0]=5
    if good:
        logits[:,1,2,:]=8;logits[:,2,0,0]=8
    targets=torch.zeros((1,5,5),dtype=torch.long);targets[:,2,:]=1;mask=torch.ones_like(targets,dtype=torch.bool);valid=torch.ones((1,1,5,5),dtype=torch.bool)
    points=torch.zeros((1,4,5,5));points[0,0,2,0]=1;points[0,1,2,4]=1
    global_conditions=torch.zeros((1,14));global_conditions[0,-3]=.04;global_conditions[0,-2]=.2
    tables=(torch.tensor((0.,1.,0.)),torch.tensor((0.,0.,1.)))
    return semantic_topology_loss(logits,targets,mask,valid,points,global_conditions,torch.ones(3),*tables,propagation_steps=5)


def test_semantic_loss_rewards_connected_conditioned_prediction() -> None:
    good,bad=_case(True),_case(False)
    assert good["reachability"]<bad["reachability"]
    assert good["condition"]<bad["condition"]
    assert good["loss"]<bad["loss"]


def test_frequency_weights_emphasize_rare_tokens_with_bounds() -> None:
    weights=frequency_weights(torch.tensor((1000,100,1,0)))
    assert weights[2]>weights[1]>=weights[0]
    assert .5<=float(weights.min()) and float(weights.max())<=5
