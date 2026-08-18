from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .contract import ModelConfig


class MobileResidual(nn.Module):
    def __init__(self, width: int, expansion: int = 2):
        super().__init__(); hidden = width * expansion
        self.expand = nn.Conv2d(width, hidden, 1, bias=False)
        self.expand_norm = nn.BatchNorm2d(hidden)
        self.depthwise = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False)
        self.depthwise_norm = nn.BatchNorm2d(hidden)
        self.project = nn.Conv2d(hidden, width, 1, bias=False)
        self.project_norm = nn.BatchNorm2d(width)
        self.scale = nn.Parameter(torch.full((1, width, 1, 1), 0.1))

    def forward(self, x):
        hidden = F.silu(self.expand_norm(self.expand(x)))
        hidden = F.silu(self.depthwise_norm(self.depthwise(hidden)))
        return x + self.scale * self.project_norm(self.project(hidden))


class SubpixelStage(nn.Module):
    def __init__(self, source: int, target: int, blocks: int):
        super().__init__()
        self.expand = nn.Conv2d(source, target * 4, 3, padding=1)
        self.blocks = nn.Sequential(*(MobileResidual(target) for _ in range(blocks)))

    def forward(self, x): return self.blocks(F.silu(F.pixel_shuffle(self.expand(x), 2)))


class MobileViewportDecoder(nn.Module):
    """GPU-oriented VAE latent decoder for the complete 256x256 viewport."""
    def __init__(self, config: ModelConfig = ModelConfig()):
        super().__init__(); self.config = config; widths = config.widths
        self.stem = nn.Conv2d(config.latent_channels, widths[0], 3, padding=1)
        self.stem_blocks = nn.Sequential(*(MobileResidual(widths[0]) for _ in range(config.residual_blocks)))
        self.stages = nn.ModuleList(SubpixelStage(widths[index], widths[index + 1], config.residual_blocks) for index in range(3))
        self.out = nn.Sequential(MobileResidual(widths[-1]), nn.Conv2d(widths[-1], 3, 3, padding=1))

    def forward(self, latent):
        if latent.shape[1:] != (self.config.latent_channels, 32, 32): raise ValueError("mobile decoder latent drifted")
        hidden = self.stem_blocks(F.silu(self.stem(latent)))
        for stage in self.stages: hidden = stage(hidden)
        return torch.sigmoid(self.out(hidden))
