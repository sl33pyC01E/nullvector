from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F

from ..maps.model import WALKABLE_TERRAIN


@torch.inference_mode()
def semantic_token_tables(codec: torch.nn.Module) -> tuple[Tensor, Tensor]:
    """Measure what every frozen VQ token decodes into, without labels or heuristics."""
    embeddings=codec.quantizer.embeddings
    quantized=embeddings[:, :, None, None]
    decoded=codec.decode(quantized)
    terrain=decoded["terrain"].float().argmax(1)
    hazard=decoded["hazard"].float().argmax(1)
    walkable=torch.zeros_like(terrain,dtype=torch.bool)
    for terrain_id in WALKABLE_TERRAIN:walkable|=terrain==int(terrain_id)
    walkable=walkable.float().mean((1,2))
    hazardous=(hazard!=0).float().mean((1,2))
    return walkable.clamp(0,1),hazardous.clamp(0,1)


def frequency_weights(counts: Tensor, *, maximum: float=5.0) -> Tensor:
    if counts.ndim!=1 or not counts.numel() or bool((counts<0).any()):raise ValueError("Token counts must be a nonnegative vector.")
    weights=(counts.sum().clamp_min(1)/(counts.clamp_min(1)*counts.numel())).sqrt()
    present=counts>0
    if bool(present.any()):weights=weights/weights[present].mean()
    return weights.clamp(.5,maximum)


def semantic_topology_loss(
    logits: Tensor,targets: Tensor,mask: Tensor,valid_mask: Tensor,point_conditions: Tensor,
    global_conditions: Tensor,token_weights: Tensor,walkable_table: Tensor,hazard_table: Tensor,
    *,propagation_steps: int|None=None,codec: torch.nn.Module|None=None,
) -> dict[str,Tensor]:
    batch,vocabulary,height,width=logits.shape
    if targets.shape!=(batch,height,width) or mask.shape!=targets.shape or valid_mask.shape!=(batch,1,height,width):raise ValueError("Semantic topology tensor shapes disagree.")
    if point_conditions.shape!=(batch,4,height,width) or global_conditions.shape!=(batch,14):raise ValueError("Semantic topology conditions disagree.")
    if token_weights.shape!=(vocabulary,) or walkable_table.shape!=(vocabulary,) or hazard_table.shape!=(vocabulary,):raise ValueError("Semantic topology token tables disagree.")
    propagation_steps=height+width if propagation_steps is None else propagation_steps
    if propagation_steps<1:raise ValueError("Propagation steps must be positive.")
    per_cell=F.cross_entropy(logits.float(),targets.long(),weight=token_weights.float(),reduction="none")
    token_loss=per_cell[mask].mean()
    probability=torch.softmax(logits.float(),dim=1)
    if codec is None:
        walk=torch.einsum("bkhw,k->bhw",probability,walkable_table.float()).clamp(1e-5,1-1e-5)
        hazard=torch.einsum("bkhw,k->bhw",probability,hazard_table.float()).clamp(1e-5,1-1e-5)
    else:
        embedded=torch.einsum("bkhw,kd->bdhw",probability,codec.quantizer.embeddings.float())
        decoded=codec.decode(embedded)
        terrain_probability=torch.softmax(decoded["terrain"].float(),dim=1);hazard_probability=torch.softmax(decoded["hazard"].float(),dim=1)
        walk_full=terrain_probability[:,list(sorted(WALKABLE_TERRAIN))].sum(1)
        hazard_full=hazard_probability[:,1:].sum(1)
        walk=F.avg_pool2d(walk_full[:,None],kernel_size=4,stride=4)[:,0].clamp(1e-5,1-1e-5)
        safe_full=-F.max_pool2d(-walk_full[:,None],kernel_size=3,stride=1,padding=1)[:,0]
        safe_walk=F.avg_pool2d(safe_full[:,None],kernel_size=4,stride=4)[:,0].clamp(1e-5,1-1e-5)
        hazard=F.avg_pool2d(hazard_full[:,None],kernel_size=4,stride=4)[:,0].clamp(1e-5,1-1e-5)
    if codec is None:safe_walk=walk
    valid=valid_mask[:,0].float();denominator=valid.sum((1,2)).clamp_min(1)
    openness=(walk*valid).sum((1,2))/denominator;hazard_fraction=(hazard*valid).sum((1,2))/denominator
    condition_loss=F.smooth_l1_loss(openness,global_conditions[:,-2].float())+F.smooth_l1_loss(hazard_fraction,global_conditions[:,-3].float())
    required=point_conditions[:,:3].amax(1).clamp(0,1)*valid
    point_loss=(-(required*walk.log()).sum((1,2))/required.sum((1,2)).clamp_min(1)).mean()
    reached=point_conditions[:,0].clamp(0,1)*safe_walk*valid
    for _ in range(propagation_steps):
        spread=F.max_pool2d(reached[:,None],3,1,1)[:,0]*safe_walk*valid
        reached=torch.maximum(reached,spread)
    destination=point_conditions[:,1:3].amax(1).clamp(0,1)*valid
    coverage=(reached*destination).sum((1,2))/destination.sum((1,2)).clamp_min(1)
    reachability_loss=(-coverage.clamp_min(1e-5).log()).mean()
    sharpness_loss=((walk*(1-walk)+hazard*(1-hazard))*valid).sum()/valid.sum().clamp_min(1)
    total=token_loss+.75*condition_loss+.5*point_loss+.2*reachability_loss+.15*sharpness_loss
    return {"loss":total,"token":token_loss,"condition":condition_loss,"point":point_loss,"reachability":reachability_loss,"sharpness":sharpness_loss,"predicted_openness":openness.mean(),"predicted_hazard":hazard_fraction.mean(),"destination_coverage":coverage.mean()}
