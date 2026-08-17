from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class Residual(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels); self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels); self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, value: Tensor) -> Tensor:
        hidden = self.conv1(F.silu(self.norm1(value)))
        return value + self.conv2(F.silu(self.norm2(hidden)))


@dataclass(slots=True)
class Refined:
    rgba: Tensor
    delta: Tensor


class NeuralCellRefiner(nn.Module):
    """Learn sub-cell coverage without replacing the parent VAE latent render."""

    def __init__(self, width: int = 64) -> None:
        super().__init__()
        self.width = width
        self.stem = nn.Conv2d(42 + 4, width, 3, padding=1)
        self.low = nn.Sequential(*(Residual(width) for _ in range(4)))
        self.up = nn.Sequential(nn.Conv2d(width, width * 4, 3, padding=1), nn.PixelShuffle(2), nn.SiLU())
        self.high_in = nn.Conv2d(width + 4, width, 3, padding=1)
        self.high = nn.Sequential(*(Residual(width) for _ in range(3)))
        self.out = nn.Conv2d(width, 4, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, living: Tensor, parent_rgba: Tensor) -> Refined:
        if living.shape[1:] != (42, 48, 48) or parent_rgba.shape[1:] != (4, 96, 96):
            raise ValueError("V7 neural refiner input geometry drifted")
        parent_low = F.avg_pool2d(parent_rgba.float(), 2)
        value = self.low(self.stem(torch.cat((living.float(), parent_low), 1)))
        value = self.high(self.high_in(torch.cat((self.up(value), parent_rgba.float()), 1)))
        delta = self.out(value)
        rgb = torch.clamp(parent_rgba[:, :3].float() + .25 * torch.tanh(delta[:, :3]), 0, 1)
        alpha = parent_rgba[:, 3:].float().clamp(1e-5, 1 - 1e-5)
        alpha = torch.sigmoid(torch.logit(alpha) + delta[:, 3:])
        return Refined(torch.cat((rgb, alpha), 1), delta)


def loss(output: Refined, target: Tensor, appendage_alpha: Tensor) -> tuple[Tensor, dict[str, float]]:
    target = target.float(); alpha = target[:, 3:]; predicted_alpha = output.rgba[:, 3:].clamp(1e-5, 1 - 1e-5)
    weight = .15 + 2.85 * alpha
    bce = (-(alpha * predicted_alpha.log() + (1 - alpha) * torch.log1p(-predicted_alpha)) * weight).sum() / weight.sum()
    intersection = (predicted_alpha * alpha).sum((1, 2, 3))
    dice = 1 - ((2 * intersection + 1) / (predicted_alpha.sum((1, 2, 3)) + alpha.sum((1, 2, 3)) + 1)).mean()
    rgb = ((output.rgba[:, :3] - target[:, :3]).abs() * weight).sum() / (weight.sum() * 3)
    edge = (predicted_alpha[:, :, 1:] - predicted_alpha[:, :, :-1] - (alpha[:, :, 1:] - alpha[:, :, :-1])).abs().mean()
    edge += (predicted_alpha[:, :, :, 1:] - predicted_alpha[:, :, :, :-1] - (alpha[:, :, :, 1:] - alpha[:, :, :, :-1])).abs().mean()
    limb = appendage_alpha.float(); limb_recall = 1 - (predicted_alpha * limb).sum() / limb.sum().clamp_min(1)
    delta_penalty = output.delta.float().square().mean()
    total = 3.0 * bce + 1.5 * dice + 2.5 * rgb + 1.25 * edge + .75 * limb_recall + 2e-4 * delta_penalty
    return total, {"loss": float(total.detach()), "alpha_bce": float(bce.detach()), "dice": float(dice.detach()), "rgb_l1": float(rgb.detach()), "edge_l1": float(edge.detach()), "limb_recall_loss": float(limb_recall.detach()), "delta_l2": float(delta_penalty.detach())}
