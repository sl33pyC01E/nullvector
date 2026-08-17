from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .contract import ModelConfig


@dataclass(slots=True)
class StudentOutput:
    latent: Tensor
    actor_state: Tensor
    hidden: Tensor
    edit_gate: Tensor


class ConvGRUCell(nn.Module):
    def __init__(self, input_channels: int, hidden: int):
        super().__init__()
        self.hidden = hidden
        self.gates = nn.Conv2d(input_channels + hidden, hidden * 2, 3, padding=1)
        self.candidate = nn.Conv2d(input_channels + hidden, hidden, 3, padding=1)

    def forward(self, value: Tensor, hidden: Tensor | None) -> Tensor:
        if hidden is None:
            hidden = value.new_zeros((len(value), self.hidden, *value.shape[-2:]))
        reset, update = torch.sigmoid(self.gates(torch.cat((value, hidden), 1))).chunk(2, 1)
        candidate = torch.tanh(self.candidate(torch.cat((value, reset * hidden), 1)))
        return (1 - update) * hidden + update * candidate


class RecurrentWorldStudent(nn.Module):
    """Compact recurrent latent transition with explicit action conditioning."""

    def __init__(self, config: ModelConfig = ModelConfig()):
        super().__init__(); self.config = config; h = config.hidden; c = config.condition
        self.action = nn.Embedding(config.action_count, c)
        self.control = nn.Sequential(nn.Linear(4, c), nn.SiLU(), nn.Linear(c, c))
        self.state = nn.Sequential(nn.Linear(64, c), nn.SiLU(), nn.Linear(c, c))
        self.actor = nn.Sequential(nn.Linear(config.actor_features, c), nn.SiLU(), nn.Linear(c, c))
        self.condition = nn.Sequential(nn.Linear(c, h), nn.SiLU(), nn.Linear(h, h))
        self.encoder = nn.Sequential(
            nn.Conv2d(config.latent_channels * 2, h, 4, 2, 1), nn.GroupNorm(16, h), nn.SiLU(),
            nn.Conv2d(h, h, 3, padding=1), nn.GroupNorm(16, h), nn.SiLU(),
        )
        self.recurrent = ConvGRUCell(h * 2, h)
        self.delta = nn.Sequential(nn.Conv2d(h, h, 3, padding=1), nn.SiLU(), nn.Conv2d(h, config.latent_channels, 3, padding=1))
        self.gate = nn.Sequential(nn.Conv2d(h, h // 2, 3, padding=1), nn.SiLU(), nn.Conv2d(h // 2, 1, 1))
        self.actor_delta = nn.Sequential(nn.Linear(h + c, h), nn.SiLU(), nn.Linear(h, config.actor_features))
        nn.init.zeros_(self.delta[-1].weight); nn.init.zeros_(self.delta[-1].bias)
        nn.init.constant_(self.gate[-1].bias, -3.0)
        nn.init.zeros_(self.actor_delta[-1].weight); nn.init.zeros_(self.actor_delta[-1].bias)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, current: Tensor, previous: Tensor, action: Tensor, control: Tensor, state: Tensor, actor_state: Tensor, hidden: Tensor | None = None) -> StudentOutput:
        condition = self.action(action) + self.control(control.float()) + self.state(state.float()) + self.actor(actor_state.float())
        condition_map = self.condition(condition)[:, :, None, None].expand(-1, -1, 16, 16)
        encoded = self.encoder(torch.cat((current, current - previous), 1))
        hidden = self.recurrent(torch.cat((encoded, condition_map), 1), hidden)
        dense = F.interpolate(hidden, scale_factor=2, mode="bilinear", align_corners=False)
        gate = torch.sigmoid(self.gate(dense))
        latent = current + gate * self.delta(dense)
        pooled = hidden.mean((2, 3))
        next_actor = actor_state + self.actor_delta(torch.cat((pooled, condition), 1))
        return StudentOutput(latent, next_actor, hidden, gate)
