from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .contract import NeuralCellMotionConfig


def phase_embedding(phase: Tensor, dimensions: int) -> Tensor:
    if phase.ndim != 1 or dimensions < 8: raise ValueError("Neural motion phase input drifted.")
    quarter = dimensions // 4; frequency = torch.arange(1, quarter + 1, device=phase.device, dtype=torch.float32)[None]
    angle = phase.float()[:, None] * math.tau * frequency
    value = torch.cat((angle.sin(), angle.cos(), (angle * .5).sin(), (angle * .5).cos()), dim=1)
    return F.pad(value, (0, dimensions - value.shape[1]))


class ConditionEncoder(nn.Module):
    def __init__(self, dimensions: int) -> None:
        super().__init__(); self.dimensions = dimensions
        self.family = nn.Embedding(5, dimensions); self.motion = nn.Embedding(13, dimensions); self.facing = nn.Embedding(8, dimensions)
        self.phase = nn.Sequential(nn.Linear(dimensions, dimensions), nn.SiLU(), nn.Linear(dimensions, dimensions))
        self.out = nn.Sequential(nn.LayerNorm(dimensions), nn.SiLU(), nn.Linear(dimensions, dimensions))

    def forward(self, family: Tensor, motion: Tensor, facing: Tensor, phase: Tensor) -> Tensor:
        batch = len(family)
        if family.shape != motion.shape or family.shape != facing.shape or family.shape != phase.shape or family.ndim != 1 or bool((family < 0).any()) or bool((family >= 5).any()) or bool((motion < 0).any()) or bool((motion >= 13).any()) or bool((facing < 0).any()) or bool((facing >= 8).any()) or not bool(torch.isfinite(phase).all()): raise ValueError("Neural motion condition drifted.")
        return self.out(self.family(family) + self.motion(motion) + self.facing(facing) + self.phase(phase_embedding(phase, self.dimensions))).reshape(batch, self.dimensions)


class MotionBlock(nn.Module):
    def __init__(self, channels: int, condition_dim: int, dropout: float) -> None:
        super().__init__(); groups = min(32, channels)
        while channels % groups: groups -= 1
        self.norm1 = nn.GroupNorm(groups, channels); self.norm2 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1); self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.film = nn.Sequential(nn.SiLU(), nn.Linear(condition_dim, channels * 2)); self.dropout = nn.Dropout2d(dropout)

    def forward(self, value: Tensor, condition: Tensor) -> Tensor:
        scale, shift = self.film(condition).chunk(2, dim=1); hidden = self.norm1(value) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        hidden = self.conv1(F.silu(hidden)); hidden = self.conv2(self.dropout(F.silu(self.norm2(hidden)))); return value + hidden


class SpatialAttention(nn.Module):
    def __init__(self, channels: int, heads: int) -> None:
        super().__init__(); self.norm = nn.LayerNorm(channels); self.attention = nn.MultiheadAttention(channels, heads, batch_first=True); self.out = nn.Linear(channels, channels)

    def forward(self, value: Tensor) -> Tensor:
        batch, channels, height, width = value.shape; tokens = value.flatten(2).transpose(1, 2); normalized = self.norm(tokens); attended, _ = self.attention(normalized, normalized, normalized, need_weights=False); return value + self.out(attended).transpose(1, 2).reshape(batch, channels, height, width)


class NeuralCellMotionUNet(nn.Module):
    """Phase-conditioned recurrent raster motion model.

    Static anatomy remains a 60-channel authority. The previous four-channel
    motion state is fed back every step, allowing cached loops to be rolled out
    without re-running an appearance model.
    """

    def __init__(self, config: NeuralCellMotionConfig = NeuralCellMotionConfig()) -> None:
        super().__init__(); self.config = config; widths = [config.base_channels * value for value in config.channel_multipliers]; condition = config.condition_dim
        self.condition = ConditionEncoder(condition); self.input = nn.Conv2d(config.static_channels + config.state_channels, widths[0], 3, padding=1)
        self.encoder0 = nn.ModuleList(MotionBlock(widths[0], condition, config.dropout) for _ in range(config.blocks_per_level)); self.down0 = nn.Conv2d(widths[0], widths[1], 4, stride=2, padding=1)
        self.encoder1 = nn.ModuleList(MotionBlock(widths[1], condition, config.dropout) for _ in range(config.blocks_per_level)); self.down1 = nn.Conv2d(widths[1], widths[2], 4, stride=2, padding=1)
        self.middle = nn.ModuleList(MotionBlock(widths[2], condition, config.dropout) for _ in range(config.blocks_per_level * 2)); self.attention = SpatialAttention(widths[2], config.attention_heads)
        self.up1 = nn.ConvTranspose2d(widths[2], widths[1], 4, stride=2, padding=1); self.merge1 = nn.Conv2d(widths[1] * 2, widths[1], 1); self.decoder1 = nn.ModuleList(MotionBlock(widths[1], condition, config.dropout) for _ in range(config.blocks_per_level))
        self.up0 = nn.ConvTranspose2d(widths[1], widths[0], 4, stride=2, padding=1); self.merge0 = nn.Conv2d(widths[0] * 2, widths[0], 1); self.decoder0 = nn.ModuleList(MotionBlock(widths[0], condition, config.dropout) for _ in range(config.blocks_per_level))
        groups = min(32, widths[0]);
        while widths[0] % groups: groups -= 1
        self.output = nn.Sequential(nn.GroupNorm(groups, widths[0]), nn.SiLU(), nn.Conv2d(widths[0], config.state_channels, 3, padding=1))

    @property
    def parameter_count(self) -> int: return sum(value.numel() for value in self.parameters())

    def forward(self, static: Tensor, previous: Tensor, family: Tensor, motion: Tensor, facing: Tensor, phase: Tensor) -> Tensor:
        if static.ndim != 4 or static.shape[1:] != (self.config.static_channels, 48, 48) or previous.shape != (len(static), self.config.state_channels, 48, 48): raise ValueError("Neural motion raster input drifted.")
        condition = self.condition(family, motion, facing, phase); x0 = self.input(torch.cat((static.float(), previous.float()), dim=1))
        for block in self.encoder0: x0 = block(x0, condition)
        x1 = self.down0(x0)
        for block in self.encoder1: x1 = block(x1, condition)
        middle = self.down1(x1)
        for block in self.middle: middle = block(middle, condition)
        middle = self.attention(middle); value = self.merge1(torch.cat((self.up1(middle), x1), dim=1))
        for block in self.decoder1: value = block(value, condition)
        value = self.merge0(torch.cat((self.up0(value), x0), dim=1))
        for block in self.decoder0: value = block(value, condition)
        raw = self.output(value); displacement = torch.tanh(raw[:, :2]); activity = torch.sigmoid(raw[:, 2:]); occupancy = static[:, :1]
        return torch.cat((displacement, activity), dim=1) * occupancy


def neural_motion_loss(predicted: Tensor, target: Tensor, previous: Tensor, static: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
    if predicted.shape != target.shape or predicted.shape != previous.shape or predicted.ndim != 4 or predicted.shape[1:] != (4, 48, 48) or static.shape != (len(predicted), 60, 48, 48): raise ValueError("Neural motion loss tensor contract drifted.")
    occupancy = static[:, :1]; visible = occupancy.sum().clamp_min(1); displacement = F.smooth_l1_loss(predicted[:, :2] * occupancy, target[:, :2] * occupancy, reduction="sum") / (visible * 2)
    activation = F.binary_cross_entropy(predicted[:, 2:3].clamp(1e-5, 1 - 1e-5), target[:, 2:3], reduction="none"); activation = (activation * occupancy).sum() / visible
    emission = F.smooth_l1_loss(predicted[:, 3:4] * occupancy, target[:, 3:4] * occupancy, reduction="sum") / visible
    horizontal = occupancy[:, :, :, 1:] * occupancy[:, :, :, :-1]; vertical = occupancy[:, :, 1:, :] * occupancy[:, :, :-1, :]
    predicted_h = predicted[:, :2, :, 1:] - predicted[:, :2, :, :-1]; target_h = target[:, :2, :, 1:] - target[:, :2, :, :-1]; predicted_v = predicted[:, :2, 1:, :] - predicted[:, :2, :-1, :]; target_v = target[:, :2, 1:, :] - target[:, :2, :-1, :]
    coherence = ((predicted_h - target_h).abs() * horizontal).sum() / (horizontal.sum().clamp_min(1) * 2) + ((predicted_v - target_v).abs() * vertical).sum() / (vertical.sum().clamp_min(1) * 2)
    temporal = F.smooth_l1_loss((predicted - previous) * occupancy, (target - previous) * occupancy, reduction="sum") / (visible * 4)
    outside = (predicted * (1 - occupancy)).abs().mean(); total = displacement + .28 * activation + .18 * emission + .22 * coherence + .12 * temporal + outside
    return total, {"loss": total.detach(), "displacement": displacement.detach(), "activation": activation.detach(), "emission": emission.detach(), "coherence": coherence.detach(), "temporal": temporal.detach(), "outside": outside.detach()}
