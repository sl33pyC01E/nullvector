from __future__ import annotations

import math

import torch
from torch import nn

from ..world_latent_dit.contract import ACTIONS, CONTROL_FEATURES, LATENT_CHANNELS, LATENT_SIZE, STATE_FEATURES
from .contract import ModelConfig


def spatial_control_fields(control: torch.Tensor, side: int = LATENT_SIZE) -> torch.Tensor:
    """Actor, aim, ray, perpendicular reach, and movement fields in camera space."""
    if control.ndim != 2 or control.shape[1] != CONTROL_FEATURES:
        raise ValueError("sparse action control must be Bx4")
    coordinate = torch.linspace(-1, 1, side, dtype=control.dtype, device=control.device)
    yy, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")
    xx = xx[None]
    yy = yy[None]
    aim_x = control[:, 2, None, None].clamp(-1, 1)
    aim_y = control[:, 3, None, None].clamp(-1, 1)
    actor = torch.exp(-(xx.square() + yy.square()) / 0.035).expand(len(control), -1, -1)
    aim = torch.exp(-((xx - aim_x).square() + (yy - aim_y).square()) / 0.025)
    length2 = (aim_x.square() + aim_y.square()).clamp_min(1e-4)
    projection = ((xx * aim_x + yy * aim_y) / length2).clamp(0, 1)
    closest_x = projection * aim_x
    closest_y = projection * aim_y
    distance2 = (xx - closest_x).square() + (yy - closest_y).square()
    ray = torch.exp(-distance2 / 0.018)
    reach = torch.exp(-distance2 / 0.08) * (projection > 0).to(control.dtype)
    movement = ((xx * control[:, 0, None, None] + yy * control[:, 1, None, None]) * 0.5 + 0.5).clamp(0, 1)
    return torch.stack((actor, aim, ray, reach, movement), dim=1)


def _modulate(value, shift, scale):
    return value * (1 + scale[:, None]) + shift[:, None]


class SparseBlock(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(width, elementwise_affine=False)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(width, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(width, width * 4), nn.GELU(), nn.Linear(width * 4, width))
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, width * 6))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, value, condition):
        shift1, scale1, gate1, shift2, scale2, gate2 = self.modulation(condition).chunk(6, 1)
        normalized = _modulate(self.norm1(value), shift1, scale1)
        value = value + gate1[:, None] * self.attention(normalized, normalized, normalized, need_weights=False)[0]
        return value + gate2[:, None] * self.mlp(_modulate(self.norm2(value), shift2, scale2))


class SparseActionDiT(nn.Module):
    """A causal latent editor that separately predicts edit support and edit content."""

    def __init__(self, config: ModelConfig = ModelConfig()):
        super().__init__()
        self.config = config
        tokens = (LATENT_SIZE // config.patch) ** 2
        self.patch = nn.Conv2d(LATENT_CHANNELS, config.width, config.patch, config.patch)
        self.spatial = nn.Conv2d(config.spatial_channels, config.width, config.patch, config.patch)
        self.position = nn.Parameter(torch.randn(1, tokens, config.width) * 0.015)
        self.action = nn.Embedding(ACTIONS, config.width)
        self.control = nn.Linear(CONTROL_FEATURES, config.width)
        self.state = nn.Linear(STATE_FEATURES, config.width)
        self.time = nn.Sequential(nn.Linear(64, config.width), nn.SiLU(), nn.Linear(config.width, config.width))
        self.blocks = nn.ModuleList(SparseBlock(config.width, config.heads) for _ in range(config.layers))
        self.norm = nn.LayerNorm(config.width, elementwise_affine=False)
        self.final_mod = nn.Sequential(nn.SiLU(), nn.Linear(config.width, config.width * 2))
        self.delta_out = nn.Linear(config.width, LATENT_CHANNELS * config.patch * config.patch)
        self.gate_out = nn.Linear(config.width, config.patch * config.patch)
        nn.init.zeros_(self.final_mod[-1].weight)
        nn.init.zeros_(self.final_mod[-1].bias)
        nn.init.zeros_(self.delta_out.weight)
        nn.init.zeros_(self.delta_out.bias)
        nn.init.zeros_(self.gate_out.weight)
        nn.init.constant_(self.gate_out.bias, config.gate_bias)

    @staticmethod
    def time_embedding(time):
        frequency = torch.exp(torch.linspace(math.log(1), math.log(10000), 32, device=time.device))
        angle = time[:, None] * frequency[None] * math.tau
        return torch.cat((torch.sin(angle), torch.cos(angle)), 1)

    def _unpatch(self, value, channels):
        batch = value.shape[0]
        patch = self.config.patch
        side = LATENT_SIZE // patch
        return value.view(batch, side, side, channels, patch, patch).permute(0, 3, 1, 4, 2, 5).reshape(batch, channels, LATENT_SIZE, LATENT_SIZE)

    def forward(self, latent, time, action, control, state):
        condition = self.time(self.time_embedding(time)) + self.action(action) + self.control(control) + self.state(state)
        field = spatial_control_fields(control, latent.shape[-1])
        token = self.patch(latent).flatten(2).transpose(1, 2) + self.spatial(field).flatten(2).transpose(1, 2) + self.position
        for block in self.blocks:
            token = block(token, condition)
        shift, scale = self.final_mod(condition).chunk(2, 1)
        token = _modulate(self.norm(token), shift, scale)
        delta = self._unpatch(self.delta_out(token), LATENT_CHANNELS)
        gate_logits = self._unpatch(self.gate_out(token), 1)
        return delta, gate_logits

    def edit(self, latent, time, action, control, state):
        delta, gate_logits = self(latent, time, action, control, state)
        gate = torch.sigmoid(gate_logits)
        return latent + gate * delta, gate, delta, gate_logits
