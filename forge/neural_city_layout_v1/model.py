from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .contract import CLASSES, CONDITION_NAMES, MASK_TOKEN, ModelConfig


def _groups(channels: int) -> int:
    groups = min(8, channels)
    while channels % groups:
        groups -= 1
    return groups


class ConditionalBlock(nn.Module):
    def __init__(self, channels: int, condition_width: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(channels), channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(_groups(channels), channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.film = nn.Linear(condition_width, channels * 2)

    def forward(self, value: Tensor, condition: Tensor) -> Tensor:
        scale, shift = self.film(condition).chunk(2, dim=1)
        hidden = self.norm1(value) * (1 + .1 * torch.tanh(scale)[:, :, None, None]) + shift[:, :, None, None]
        hidden = self.conv1(F.silu(hidden))
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return value + hidden


class NeuralCityLayout(nn.Module):
    def __init__(self, config: ModelConfig = ModelConfig()) -> None:
        super().__init__()
        self.config = config
        channels = [config.width * 2 ** level for level in range(config.levels)]
        condition_width = config.width * 2
        self.embedding = nn.Embedding(len(CLASSES) + 1, config.width)
        self.coordinate_projection = nn.Conv2d(2, config.width, 1)
        self.condition = nn.Sequential(nn.Linear(len(CONDITION_NAMES), condition_width), nn.SiLU(), nn.Linear(condition_width, condition_width))
        self.input = nn.Conv2d(config.width, channels[0], 3, padding=1)
        self.down_blocks = nn.ModuleList()
        self.downsample = nn.ModuleList()
        for level, channel in enumerate(channels):
            self.down_blocks.append(nn.ModuleList(ConditionalBlock(channel, condition_width) for _ in range(config.blocks_per_level)))
            if level + 1 < len(channels): self.downsample.append(nn.Conv2d(channel, channels[level + 1], 4, stride=2, padding=1))
        self.global_mix = nn.Sequential(nn.Conv2d(channels[-1] * 2, channels[-1], 1), nn.SiLU(), nn.Conv2d(channels[-1], channels[-1], 3, padding=1))
        self.up_projection = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for level in range(len(channels) - 2, -1, -1):
            self.up_projection.append(nn.Conv2d(channels[level + 1] + channels[level], channels[level], 1))
            self.up_blocks.append(nn.ModuleList(ConditionalBlock(channels[level], condition_width) for _ in range(config.blocks_per_level)))
        self.output = nn.Sequential(nn.GroupNorm(_groups(channels[0]), channels[0]), nn.SiLU(), nn.Conv2d(channels[0], len(CLASSES), 1))

    def forward(self, tokens: Tensor, conditions: Tensor) -> Tensor:
        if tokens.ndim != 3 or tokens.dtype != torch.long or bool(((tokens < 0) | (tokens > MASK_TOKEN)).any()):
            raise ValueError("City tokens have invalid shape/dtype/vocabulary.")
        if conditions.shape != (tokens.shape[0], len(CONDITION_NAMES)) or not bool(torch.isfinite(conditions).all()):
            raise ValueError("City conditions have invalid shape or values.")
        condition = self.condition(conditions.float())
        height, width = tokens.shape[-2:]
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, height, device=tokens.device, dtype=torch.float32),
            torch.linspace(-1, 1, width, device=tokens.device, dtype=torch.float32),
            indexing="ij",
        )
        coordinates = torch.stack((xx, yy))[None].expand(tokens.shape[0], -1, -1, -1)
        hidden = self.input(self.embedding(tokens).permute(0, 3, 1, 2) + self.coordinate_projection(coordinates))
        skips = []
        for level, blocks in enumerate(self.down_blocks):
            for block in blocks: hidden = block(hidden, condition)
            skips.append(hidden)
            if level < len(self.downsample): hidden = self.downsample[level](hidden)
        global_mean = hidden.mean(dim=(2, 3), keepdim=True).expand_as(hidden)
        hidden = hidden + self.global_mix(torch.cat((hidden, global_mean), dim=1))
        for index, (projection, blocks) in enumerate(zip(self.up_projection, self.up_blocks, strict=True)):
            skip = skips[-2 - index]
            hidden = F.interpolate(hidden, size=skip.shape[-2:], mode="nearest")
            hidden = projection(torch.cat((hidden, skip), dim=1))
            for block in blocks: hidden = block(hidden, condition)
        return self.output(hidden)


def build_model(config: ModelConfig, *, seed: int | None = None) -> NeuralCityLayout:
    previous = torch.get_rng_state()
    try:
        torch.manual_seed(config.seed if seed is None else seed)
        return NeuralCityLayout(config)
    finally:
        torch.set_rng_state(previous)


@torch.inference_mode()
def sample_layout(model: nn.Module, conditions: Tensor, *, steps: int = 12) -> Tensor:
    if not 1 <= steps <= 64:
        raise ValueError("City sampler step count drifted.")
    batch = conditions.shape[0]
    tokens = torch.full((batch, 64, 64), MASK_TOKEN, dtype=torch.long, device=conditions.device)
    for iteration in range(steps):
        remaining = tokens == MASK_TOKEN
        logits = model(tokens, conditions)
        probability = torch.softmax(logits.float(), dim=1)
        confidence, proposal = probability.max(dim=1)
        for index in range(batch):
            candidates = torch.nonzero(remaining[index].flatten(), as_tuple=False).flatten()
            if not candidates.numel(): continue
            target = math.ceil(tokens[index].numel() * (iteration + 1) / steps)
            reveal = max(1, min(int(candidates.numel()), target - (tokens[index].numel() - int(candidates.numel()))))
            score = confidence[index].flatten().index_select(0, candidates)
            chosen = candidates.index_select(0, torch.argsort(score, descending=True, stable=True)[:reveal])
            tokens[index].flatten()[chosen] = proposal[index].flatten()[chosen]
    if bool((tokens == MASK_TOKEN).any()):
        raise RuntimeError("City sampler left masked tokens.")
    return tokens
