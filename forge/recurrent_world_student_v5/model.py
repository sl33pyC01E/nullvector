from __future__ import annotations

import torch
from torch import Tensor, nn

from ..recurrent_world_student_v3.model import RecurrentWorldStudent
from ..world_latent_dit.contract import ModelConfig


class PerceptionAdapter(nn.Module):
    """Spatial sight/map-memory adapter with a compact global state projection."""

    def __init__(self):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1), nn.SiLU(),
            nn.Conv2d(16, 48, 1),
        )
        self.summary = nn.Sequential(nn.Linear(6, 64), nn.SiLU(), nn.Linear(64, 64))
        nn.init.zeros_(self.spatial[-1].weight)
        nn.init.zeros_(self.spatial[-1].bias)
        nn.init.zeros_(self.summary[-1].weight)
        nn.init.zeros_(self.summary[-1].bias)

    def forward(self, visibility: Tensor, memory: Tensor):
        perception = torch.cat((visibility.float(), memory.float()), 1)
        newly_seen = torch.clamp(visibility.float() - memory.float(), min=0)
        overlap = visibility.float() * memory.float()
        summary = torch.cat((
            visibility.float().mean((2, 3)), memory.float().mean((2, 3)),
            newly_seen.mean((2, 3)), overlap.mean((2, 3)),
            visibility.float().amax((2, 3)), memory.float().amax((2, 3)),
        ), 1)
        return self.spatial(perception), self.summary(summary)


class PerceptionRecurrentWorldStudent(nn.Module):
    def __init__(self, config: ModelConfig = ModelConfig()):
        super().__init__()
        self.world = RecurrentWorldStudent(config)
        self.perception = PerceptionAdapter()

    @property
    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())

    def load_parent(self, state):
        self.world.load_state_dict(state, strict=True)

    def action(self, current, previous, action, control, state, actor_state, visibility, memory):
        spatial, summary = self.perception(visibility, memory)
        return self.world.action(current + spatial, previous + spatial, action, control, state + summary, actor_state)

    def actor(self, current, previous, action, control, state, visibility, memory):
        _, summary = self.perception(visibility, memory)
        return self.world.actor(current, previous, action, control, state + summary)
