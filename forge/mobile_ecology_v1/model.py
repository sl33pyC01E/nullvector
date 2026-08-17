from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ..nature_behavior_nn.contract import MAX_NEIGHBORS, NEIGHBOR_FEATURES, RESOURCE_FEATURES, SELF_FEATURES
from ..nature_sim_v2.contract import INTENTS, RESOURCE_NAMES
from .contract import MobileEcologyConfig


@dataclass(slots=True)
class MobileEcologyOutput:
    intent_logits: Tensor
    direction: Tensor
    urgency: Tensor


class MobileEcologyPolicy(nn.Module):
    """Compact organism policy for batched mobile ecology decisions."""
    def __init__(self, config: MobileEcologyConfig = MobileEcologyConfig()) -> None:
        super().__init__(); self.config = config; sw = config.self_width; tw = config.token_width; hw = config.hidden_width
        self.self_encoder = nn.Sequential(nn.Linear(SELF_FEATURES, sw), nn.LayerNorm(sw), nn.SiLU(), nn.Linear(sw, sw))
        resource_features = len(RESOURCE_NAMES) * RESOURCE_FEATURES
        neighbor_features = MAX_NEIGHBORS * NEIGHBOR_FEATURES + MAX_NEIGHBORS
        self.resource_encoder = nn.Sequential(nn.LayerNorm(resource_features), nn.Linear(resource_features, tw), nn.SiLU(), nn.Linear(tw, tw))
        self.neighbor_encoder = nn.Sequential(nn.LayerNorm(neighbor_features), nn.Linear(neighbor_features, tw), nn.SiLU(), nn.Linear(tw, tw))
        self.trunk = nn.Sequential(nn.Linear(sw + tw * 2, hw), nn.LayerNorm(hw), nn.SiLU(), nn.Linear(hw, hw), nn.SiLU())
        self.intent = nn.Linear(hw, len(INTENTS)); self.direction = nn.Sequential(nn.Linear(hw, 2), nn.Tanh()); self.urgency = nn.Linear(hw, 1)

    @property
    def parameter_count(self) -> int: return sum(value.numel() for value in self.parameters())

    def forward(self, self_features: Tensor, resource: Tensor, neighbor: Tensor, neighbor_mask: Tensor) -> MobileEcologyOutput:
        if self_features.shape[-1] != SELF_FEATURES or resource.shape[-2:] != (len(RESOURCE_NAMES), RESOURCE_FEATURES) or neighbor.shape[-2:] != (MAX_NEIGHBORS, NEIGHBOR_FEATURES): raise ValueError("mobile ecology input geometry drifted")
        self_token = self.self_encoder(self_features.float()); resource_token = self.resource_encoder(resource.float().flatten(1))
        active = neighbor_mask.bool(); masked_neighbor = neighbor.float() * active[:, :, None]
        neighbor_token = self.neighbor_encoder(torch.cat((masked_neighbor.flatten(1), active.float()), 1))
        hidden = self.trunk(torch.cat((self_token, resource_token, neighbor_token), 1))
        return MobileEcologyOutput(self.intent(hidden), self.direction(hidden), self.urgency(hidden)[:, 0])


class MobileEcologyGraph(nn.Module):
    def __init__(self, model: MobileEcologyPolicy) -> None: super().__init__(); self.model = model
    def forward(self, self_features: Tensor, resource: Tensor, neighbor: Tensor, neighbor_mask: Tensor):
        value = self.model(self_features, resource, neighbor, neighbor_mask > .5)
        return value.intent_logits, value.direction, value.urgency
