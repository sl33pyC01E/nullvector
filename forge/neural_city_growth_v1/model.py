from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ..neural_city_layout_v1.contract import CLASSES
from ..neural_city_layout_v1.model import ConditionalBlock, _groups
from .contract import GROWTH_CONDITION_NAMES, SITE_X_INDEX, SITE_Y_INDEX, GrowthModelConfig


class NeuralCityGrowth(nn.Module):
    def __init__(self, config: GrowthModelConfig = GrowthModelConfig()) -> None:
        super().__init__(); self.config = config
        channels = [config.width * 2 ** level for level in range(config.levels)]; condition_width = config.width * 2
        self.embedding = nn.Embedding(len(CLASSES), config.width); self.coordinate_projection = nn.Conv2d(5, config.width, 1)
        self.condition = nn.Sequential(nn.Linear(len(GROWTH_CONDITION_NAMES), condition_width), nn.SiLU(), nn.Linear(condition_width, condition_width))
        self.input = nn.Conv2d(config.width, channels[0], 3, padding=1); self.down_blocks = nn.ModuleList(); self.downsample = nn.ModuleList()
        for level, channel in enumerate(channels):
            self.down_blocks.append(nn.ModuleList(ConditionalBlock(channel, condition_width) for _ in range(config.blocks_per_level)))
            if level + 1 < len(channels): self.downsample.append(nn.Conv2d(channel, channels[level + 1], 4, stride=2, padding=1))
        self.global_mix = nn.Sequential(nn.Conv2d(channels[-1] * 2, channels[-1], 1), nn.SiLU(), nn.Conv2d(channels[-1], channels[-1], 3, padding=1))
        self.up_projection = nn.ModuleList(); self.up_blocks = nn.ModuleList()
        for level in range(len(channels) - 2, -1, -1):
            self.up_projection.append(nn.Conv2d(channels[level + 1] + channels[level], channels[level], 1)); self.up_blocks.append(nn.ModuleList(ConditionalBlock(channels[level], condition_width) for _ in range(config.blocks_per_level)))
        self.output = nn.Sequential(nn.GroupNorm(_groups(channels[0]), channels[0]), nn.SiLU(), nn.Conv2d(channels[0], len(CLASSES), 1))

    def forward(self, current: Tensor, conditions: Tensor) -> Tensor:
        if current.ndim != 3 or current.dtype != torch.long or bool(((current < 0) | (current >= len(CLASSES))).any()): raise ValueError("Growth current tokens drifted.")
        if conditions.shape != (current.shape[0], len(GROWTH_CONDITION_NAMES)) or not bool(torch.isfinite(conditions).all()): raise ValueError("Growth conditions drifted.")
        height, width = current.shape[-2:]; yy, xx = torch.meshgrid(torch.linspace(0, 1, height, device=current.device), torch.linspace(0, 1, width, device=current.device), indexing="ij"); site_x = conditions[:, SITE_X_INDEX, None, None]; site_y = conditions[:, SITE_Y_INDEX, None, None]; dx = xx[None] - site_x; dy = yy[None] - site_y; heat = torch.exp(-(dx.square() + dy.square()) / .025); coordinates = torch.stack((xx * 2 - 1, yy * 2 - 1))[None].expand(current.shape[0], -1, -1, -1); spatial = torch.cat((coordinates, dx[:, None], dy[:, None], heat[:, None]), 1)
        condition = self.condition(conditions.float()); hidden = self.input(self.embedding(current).permute(0, 3, 1, 2) + self.coordinate_projection(spatial)); skips = []
        for level, blocks in enumerate(self.down_blocks):
            for block in blocks: hidden = block(hidden, condition)
            skips.append(hidden)
            if level < len(self.downsample): hidden = self.downsample[level](hidden)
        mean = hidden.mean((2, 3), keepdim=True).expand_as(hidden); hidden = hidden + self.global_mix(torch.cat((hidden, mean), 1))
        for index, (projection, blocks) in enumerate(zip(self.up_projection, self.up_blocks, strict=True)):
            skip = skips[-2 - index]; hidden = F.interpolate(hidden, size=skip.shape[-2:], mode="nearest"); hidden = projection(torch.cat((hidden, skip), 1))
            for block in blocks: hidden = block(hidden, condition)
        return self.output(hidden)


def build_model(config: GrowthModelConfig, *, seed: int | None = None) -> NeuralCityGrowth:
    previous = torch.get_rng_state()
    try: torch.manual_seed(config.seed if seed is None else seed); return NeuralCityGrowth(config)
    finally: torch.set_rng_state(previous)
