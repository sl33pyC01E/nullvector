from __future__ import annotations

import torch
from torch import Tensor

from ..creature_stage_developmental_motion.contract import MAX_DISPLACEMENT
from ..creature_stage_developmental_actuator_v2.contract import CausalActuatorConfig
from ..creature_stage_developmental_actuator_v2.model import (
    MuscleCausalCellularActuator,
    MuscleCausalSkeletonActuator,
)
from .contract import BoneProjectionConfig


def project_bone_lengths(
    node_state: Tensor,
    node_features: Tensor,
    node_mask: Tensor,
    node_adjacency: Tensor,
    config: BoneProjectionConfig,
) -> Tensor:
    """Parallel differentiable distance projection; preserves node zero exactly."""

    rest = node_features[:, :, :2].float() * 16.0
    positions = rest + node_state[:, :, :2].float() * MAX_DISPLACEMENT
    identity = torch.eye(positions.shape[1], dtype=torch.bool, device=positions.device)[None]
    edge = node_adjacency & ~identity & node_mask[:, :, None] & node_mask[:, None, :]
    active = edge[:, :, :, None].to(positions.dtype)
    rest_delta = rest[:, None, :, :] - rest[:, :, None, :]
    rest_length = rest_delta.norm(dim=3, keepdim=True)
    anchor = positions[:, :1].clone()
    for _ in range(config.iterations):
        delta = positions[:, None, :, :] - positions[:, :, None, :]
        distance = delta.norm(dim=3, keepdim=True).clamp_min(1e-6)
        correction = delta * ((distance - rest_length) / distance) * active
        degree = active.sum(dim=2).clamp_min(1.0)
        positions = positions + correction.sum(dim=2) / degree * (config.relaxation * .5)
        positions = torch.cat((anchor, positions[:, 1:]), dim=1)
    displacement = (positions - rest) / MAX_DISPLACEMENT
    result = torch.cat((displacement, node_state[:, :, 2:].float()), dim=2)
    return result * node_mask[:, :, None].to(result.dtype)


class LengthProjectedSkeletonActuator(MuscleCausalSkeletonActuator):
    def __init__(self, config: CausalActuatorConfig, projection: BoneProjectionConfig) -> None:
        super().__init__(config)
        self.projection = projection

    def forward(self, *args, **kwargs):
        node_state, muscle_activation, force = super().forward(*args, **kwargs)
        node_features, _, node_mask, node_adjacency = args[:4]
        node_state = project_bone_lengths(
            node_state, node_features, node_mask, node_adjacency, self.projection,
        )
        return node_state, muscle_activation, force


class LengthProjectedCellularActuator(MuscleCausalCellularActuator):
    def __init__(
        self,
        config: CausalActuatorConfig | None = None,
        projection: BoneProjectionConfig | None = None,
    ) -> None:
        super().__init__(config or CausalActuatorConfig())
        self.actuator = LengthProjectedSkeletonActuator(self.config, projection or BoneProjectionConfig())
