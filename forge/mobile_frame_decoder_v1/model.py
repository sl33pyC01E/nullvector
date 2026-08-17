from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .contract import MobileDecoderConfig


class SeparableResidual(nn.Module):
    def __init__(self, source: int, target: int) -> None:
        super().__init__()
        self.project = nn.Conv2d(source, target, 1)
        self.depthwise = nn.Conv2d(target, target, 3, padding=1, groups=target)
        self.pointwise = nn.Conv2d(target, target, 1)
        self.norm = nn.GroupNorm(max(1, min(8, target // 8)), target)

    def forward(self, value: Tensor) -> Tensor:
        value = self.project(value)
        return value + self.pointwise(F.silu(self.norm(self.depthwise(value))))


class MobileFrameDecoder(nn.Module):
    """Low-memory latent rasterizer specialized for arm64 mobile inference."""

    def __init__(self, config: MobileDecoderConfig = MobileDecoderConfig()) -> None:
        super().__init__(); self.config = config; widths = config.widths
        self.input = nn.Conv2d(config.latent_channels, widths[0], 1)
        self.blocks = nn.ModuleList(SeparableResidual(widths[index], widths[index + 1]) for index in range(3))
        self.refine = nn.Sequential(SeparableResidual(widths[-1], widths[-1]), nn.Conv2d(widths[-1], 3, 3, padding=1))

    def forward(self, latent: Tensor) -> Tensor:
        if latent.ndim != 4 or latent.shape[1:] != (self.config.latent_channels, 32, 32):
            raise ValueError("Mobile decoder latent shape drifted.")
        hidden = self.input(latent.float())
        for block in self.blocks:
            hidden = block(F.interpolate(hidden, scale_factor=2, mode="nearest"))
        return torch.sigmoid(self.refine(hidden))


def build_model(config: MobileDecoderConfig, *, seed: int = 0x4D4F42494C454D44) -> MobileFrameDecoder:
    previous = torch.get_rng_state()
    try: torch.manual_seed(seed); return MobileFrameDecoder(config)
    finally: torch.set_rng_state(previous)
