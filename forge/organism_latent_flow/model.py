from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .contract import OrganismFlowConfig


def timestep_embedding(time: Tensor, dimensions: int) -> Tensor:
    if time.ndim != 1 or dimensions < 4:
        raise ValueError("Organism flow timestep input drifted.")
    half = dimensions // 2
    frequencies = torch.exp(-math.log(10_000) * torch.arange(half, device=time.device, dtype=torch.float32) / max(half - 1, 1))
    value = time.float()[:, None] * frequencies[None] * 1000
    embedding = torch.cat((value.sin(), value.cos()), dim=1)
    return F.pad(embedding, (0, dimensions - embedding.shape[1]))


class FlowBlock(nn.Module):
    def __init__(self, channels: int, condition_dim: int) -> None:
        super().__init__()
        groups = min(32, channels)
        while channels % groups:
            groups -= 1
        self.norm1 = nn.GroupNorm(groups, channels)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.film = nn.Sequential(nn.SiLU(), nn.Linear(condition_dim, channels * 2))
        squeezed = max(16, channels // 8)
        self.channel_attention = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, squeezed, 1), nn.SiLU(), nn.Conv2d(squeezed, channels, 1), nn.Sigmoid())

    def forward(self, value: Tensor, condition: Tensor) -> Tensor:
        scale, shift = self.film(condition).chunk(2, dim=1)
        hidden = self.norm1(value) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        hidden = self.conv1(F.silu(hidden))
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return value + hidden * self.channel_attention(hidden)


class HierarchicalOrganismFlow(nn.Module):
    def __init__(self, config: OrganismFlowConfig = OrganismFlowConfig()) -> None:
        super().__init__()
        self.config = config
        cw, fw, cd = config.coarse_width, config.fine_width, config.condition_dim
        self.time = nn.Sequential(nn.Linear(config.time_dim, cd), nn.SiLU(), nn.Linear(cd, cd))
        self.condition = nn.Sequential(nn.Linear(192, cd), nn.SiLU(), nn.Linear(cd, cd))
        self.coarse_in = nn.Conv2d(config.coarse_channels + config.fine_channels, cw, 3, padding=1)
        self.fine_in = nn.Conv2d(config.fine_channels, fw, 3, padding=1)
        self.coarse_blocks = nn.ModuleList(FlowBlock(cw, cd) for _ in range(config.depth))
        self.fine_blocks = nn.ModuleList(FlowBlock(fw, cd) for _ in range(config.depth))
        self.fine_to_coarse = nn.ModuleList(nn.Conv2d(fw, cw, 1) for _ in range(config.depth))
        self.coarse_to_fine = nn.ModuleList(nn.Conv2d(cw, fw, 1) for _ in range(config.depth))
        self.coarse_out = nn.Sequential(nn.GroupNorm(32, cw), nn.SiLU(), nn.Conv2d(cw, config.coarse_channels, 3, padding=1))
        fine_groups = min(32, fw)
        while fw % fine_groups:
            fine_groups -= 1
        self.fine_out = nn.Sequential(nn.GroupNorm(fine_groups, fw), nn.SiLU(), nn.Conv2d(fw, config.fine_channels, 3, padding=1))

    def conditioning(self, condition: Tensor, time: Tensor, keep_condition: Tensor | None = None) -> Tensor:
        if condition.ndim != 2 or condition.shape[1] != 192 or time.shape != (len(condition),):
            raise ValueError("Organism flow conditioning drifted.")
        embedded = self.condition(condition.float())
        if keep_condition is not None:
            if keep_condition.shape != (len(condition),):
                raise ValueError("Organism flow condition mask drifted.")
            embedded = embedded * keep_condition[:, None].float()
        return embedded + self.time(timestep_embedding(time, self.config.time_dim))

    def forward(self, coarse: Tensor, fine: Tensor, time: Tensor, condition: Tensor, keep_condition: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if coarse.ndim != 4 or coarse.shape[1:] != (32, 12, 12) or fine.ndim != 4 or fine.shape[1:] != (16, 24, 24) or len(coarse) != len(fine):
            raise ValueError("Organism flow latent pyramid drifted.")
        modulation = self.conditioning(condition, time, keep_condition)
        fine_value = self.fine_in(fine.float())
        coarse_value = self.coarse_in(torch.cat((coarse.float(), F.avg_pool2d(fine.float(), 2)), dim=1))
        for coarse_block, fine_block, down, up in zip(self.coarse_blocks, self.fine_blocks, self.fine_to_coarse, self.coarse_to_fine, strict=True):
            coarse_value = coarse_block(coarse_value + down(F.avg_pool2d(fine_value, 2)), modulation)
            fine_value = fine_block(fine_value + F.interpolate(up(coarse_value), scale_factor=2, mode="nearest"), modulation)
        return self.coarse_out(coarse_value), self.fine_out(fine_value)


@torch.no_grad()
def integrate_flow(model: HierarchicalOrganismFlow, coarse_noise: Tensor, fine_noise: Tensor, condition: Tensor, *, steps: int = 32, guidance: float = 1.6) -> tuple[Tensor, Tensor]:
    if type(steps) is not int or not 4 <= steps <= 128 or not math.isfinite(guidance) or not 0 <= guidance <= 5:
        raise ValueError("Organism flow integration contract drifted.")
    coarse, fine = coarse_noise.float().clone(), fine_noise.float().clone()
    keep = torch.ones(len(condition), device=condition.device)
    drop = torch.zeros(len(condition), device=condition.device)
    dt = 1.0 / steps
    model.eval()
    for index in range(steps):
        time = torch.full((len(condition),), (index + .5) / steps, device=condition.device)
        conditional_coarse, conditional_fine = model(coarse, fine, time, condition, keep)
        if guidance == 1:
            velocity_coarse, velocity_fine = conditional_coarse, conditional_fine
        else:
            unconditional_coarse, unconditional_fine = model(coarse, fine, time, condition, drop)
            velocity_coarse = unconditional_coarse + guidance * (conditional_coarse - unconditional_coarse)
            velocity_fine = unconditional_fine + guidance * (conditional_fine - unconditional_fine)
        coarse = coarse + dt * velocity_coarse.float()
        fine = fine + dt * velocity_fine.float()
    return coarse, fine


def flow_matching_loss(model: HierarchicalOrganismFlow, target_coarse: Tensor, target_fine: Tensor, condition: Tensor, time: Tensor, noise_coarse: Tensor, noise_fine: Tensor, keep_condition: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
    blend_coarse = noise_coarse * (1 - time[:, None, None, None]) + target_coarse * time[:, None, None, None]
    blend_fine = noise_fine * (1 - time[:, None, None, None]) + target_fine * time[:, None, None, None]
    target_velocity_coarse = target_coarse - noise_coarse
    target_velocity_fine = target_fine - noise_fine
    predicted_coarse, predicted_fine = model(blend_coarse, blend_fine, time, condition, keep_condition)
    coarse_mse = F.mse_loss(predicted_coarse.float(), target_velocity_coarse.float())
    fine_mse = F.mse_loss(predicted_fine.float(), target_velocity_fine.float())
    estimate_coarse = blend_coarse + (1 - time[:, None, None, None]) * predicted_coarse.float()
    estimate_fine = blend_fine + (1 - time[:, None, None, None]) * predicted_fine.float()
    endpoint_l1 = F.l1_loss(estimate_coarse, target_coarse) + F.l1_loss(estimate_fine, target_fine)
    total = coarse_mse + fine_mse + .12 * endpoint_l1
    return total, {"loss": total.detach(), "coarse_mse": coarse_mse.detach(), "fine_mse": fine_mse.detach(), "endpoint_l1": endpoint_l1.detach()}
