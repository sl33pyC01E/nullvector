from __future__ import annotations

import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ..map_topology_neural.contract import GLOBAL_CONDITION_NAMES
from ..maps.model import THEMES
from .conditioning import CONDITION_CHANNELS, build_spatial_conditions
from .contract import CODEBOOK_SIZE, MASK_TOKEN, PriorV2Config


def _groups(channels: int) -> int:
    groups = min(8, channels)
    while channels % groups:
        groups -= 1
    return groups


class FiLMResidual(nn.Module):
    def __init__(self, channels: int, condition_width: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(channels), channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(_groups(channels), channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.film = nn.Linear(condition_width, channels * 2)

    def forward(self, value: Tensor, valid: Tensor, condition: Tensor) -> Tensor:
        scale, shift = self.film(condition).chunk(2, dim=1)
        hidden = self.norm1(value)
        hidden = hidden * (1.0 + 0.1 * torch.tanh(scale)[:, :, None, None]) + shift[:, :, None, None]
        hidden = self.conv1(F.silu(hidden)) * valid
        hidden = self.conv2(F.silu(self.norm2(hidden))) * valid
        return (value + hidden) * valid


class AxialGlobalContext(nn.Module):
    """Give every bottleneck cell direct row, column, and map-wide context."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.mix = nn.Conv2d(channels * 4, channels, 1)
        self.norm = nn.GroupNorm(_groups(channels), channels)
        self.output = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, value: Tensor, valid: Tensor) -> Tensor:
        weights = valid.to(value.dtype)
        row = (value * weights).sum(dim=3, keepdim=True) / weights.sum(dim=3, keepdim=True).clamp_min(1)
        col = (value * weights).sum(dim=2, keepdim=True) / weights.sum(dim=2, keepdim=True).clamp_min(1)
        global_mean = (value * weights).sum(dim=(2, 3), keepdim=True) / weights.sum(dim=(2, 3), keepdim=True).clamp_min(1)
        mixed = torch.cat((
            value,
            row.expand_as(value),
            col.expand_as(value),
            global_mean.expand_as(value),
        ), dim=1)
        return (value + self.output(F.silu(self.norm(self.mix(mixed))))) * weights


class MultiScaleTopologyPrior(nn.Module):
    """Multi-scale masked-token prior with explicit whole-map receptive field."""

    def __init__(self, config: PriorV2Config) -> None:
        super().__init__()
        self.config = config
        channels = [config.width * (2 ** level) for level in range(config.levels)]
        condition_width = config.width * 2
        self.token_embedding = nn.Embedding(CODEBOOK_SIZE + 1, config.width)
        self.spatial_projection = nn.Conv2d(len(CONDITION_CHANNELS), config.width, 1)
        self.theme_embedding = nn.Embedding(len(THEMES), condition_width)
        self.global_projection = nn.Linear(len(GLOBAL_CONDITION_NAMES), condition_width)
        self.mask_projection = nn.Linear(1, condition_width)
        self.input = nn.Conv2d(config.width, channels[0], 3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.downsample = nn.ModuleList()
        for level, channel in enumerate(channels):
            self.down_blocks.append(nn.ModuleList(
                FiLMResidual(channel, condition_width) for _ in range(config.blocks_per_level)
            ))
            if level + 1 < len(channels):
                self.downsample.append(nn.Conv2d(channel, channels[level + 1], 3, stride=2, padding=1))

        self.global_context = AxialGlobalContext(channels[-1])
        self.up_projection = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for level in range(len(channels) - 2, -1, -1):
            self.up_projection.append(nn.Conv2d(channels[level + 1] + channels[level], channels[level], 1))
            self.up_blocks.append(nn.ModuleList(
                FiLMResidual(channels[level], condition_width) for _ in range(config.blocks_per_level)
            ))
        self.output_norm = nn.GroupNorm(_groups(channels[0]), channels[0])
        self.output = nn.Conv2d(channels[0], CODEBOOK_SIZE, 1)

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        required = {"tokens", "valid_mask", "point_conditions", "global_conditions", "theme_index", "mask_fraction"}
        if set(batch) != required:
            raise ValueError("Prior-v2 batch members drifted.")
        tokens = batch["tokens"]
        valid = batch["valid_mask"]
        points = batch["point_conditions"]
        global_conditions = batch["global_conditions"]
        themes = batch["theme_index"]
        mask_fraction = batch["mask_fraction"]
        if tokens.ndim != 3 or tokens.dtype != torch.long:
            raise ValueError("Prior-v2 tokens must be int64 B,H,W.")
        batch_size, height, width = tokens.shape
        if bool(((tokens < 0) | (tokens > MASK_TOKEN)).any()):
            raise ValueError("Prior-v2 tokens exceed the codebook plus mask vocabulary.")
        if valid.shape != (batch_size, 1, height, width) or valid.dtype != torch.bool:
            raise ValueError("Prior-v2 valid-mask shape/dtype drifted.")
        if points.shape != (batch_size, 4, height, width):
            raise ValueError("Prior-v2 point-condition shape drifted.")
        if global_conditions.shape != (batch_size, len(GLOBAL_CONDITION_NAMES)):
            raise ValueError("Prior-v2 global-condition shape drifted.")
        if themes.shape != (batch_size,) or themes.dtype != torch.long:
            raise ValueError("Prior-v2 theme condition drifted.")
        if mask_fraction.shape != (batch_size, 1):
            raise ValueError("Prior-v2 mask-fraction shape drifted.")
        if bool(((themes < 0) | (themes >= len(THEMES))).any()):
            raise ValueError("Prior-v2 theme index is outside vocabulary.")
        if not bool(torch.isfinite(global_conditions).all()) or not bool(torch.isfinite(mask_fraction).all()):
            raise ValueError("Prior-v2 scalar conditions contain non-finite values.")

        condition = (
            self.theme_embedding(themes)
            + self.global_projection(global_conditions.float())
            + self.mask_projection(mask_fraction.float())
        )
        valid_float = valid.to(torch.float32)
        spatial = build_spatial_conditions(points, valid)
        hidden = self.token_embedding(tokens).permute(0, 3, 1, 2)
        hidden = self.input(hidden + self.spatial_projection(spatial)) * valid_float
        skips: list[tuple[Tensor, Tensor]] = []
        current_valid = valid_float
        for level, blocks in enumerate(self.down_blocks):
            for block in blocks:
                hidden = block(hidden, current_valid, condition)
            skips.append((hidden, current_valid))
            if level < len(self.downsample):
                hidden = self.downsample[level](hidden)
                current_valid = F.interpolate(current_valid, size=hidden.shape[-2:], mode="nearest")
                hidden *= current_valid
        hidden = self.global_context(hidden, current_valid)
        for index, (projection, blocks) in enumerate(zip(self.up_projection, self.up_blocks, strict=True)):
            skip, skip_valid = skips[-2 - index]
            hidden = F.interpolate(hidden, size=skip.shape[-2:], mode="nearest")
            hidden = projection(torch.cat((hidden, skip), dim=1)) * skip_valid
            for block in blocks:
                hidden = block(hidden, skip_valid, condition)
        return self.output(F.silu(self.output_norm(hidden)))


def build_prior_v2(config: PriorV2Config, *, init_seed: int | None = None) -> MultiScaleTopologyPrior:
    seed = config.seed if init_seed is None else int(init_seed)
    if not 0 <= seed < 1 << 63:
        raise ValueError("Prior-v2 initialization seed must be unsigned 63-bit.")
    previous = torch.get_rng_state()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    try:
        torch.set_rng_state(generator.get_state())
        model = MultiScaleTopologyPrior(config)
    finally:
        torch.set_rng_state(previous)
    return model.to(device="cpu", dtype=torch.float32)


def masked_token_loss_v2(logits: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
    if logits.ndim != 4 or logits.shape[1] != CODEBOOK_SIZE:
        raise ValueError("Prior-v2 logits have the wrong vocabulary.")
    if targets.shape != logits.shape[:1] + logits.shape[2:] or mask.shape != targets.shape:
        raise ValueError("Prior-v2 loss shapes disagree.")
    if mask.dtype != torch.bool or not bool(mask.any()):
        raise ValueError("Prior-v2 loss requires masked targets.")
    loss = F.cross_entropy(logits.float(), targets.long(), reduction="none")
    return loss[mask].mean()


def sample_parallel_v2(
    model: nn.Module,
    conditions: dict[str, Tensor],
    *,
    sampling_steps: int,
) -> dict[str, Tensor]:
    valid = conditions["valid_mask"][:, 0]
    tokens = torch.zeros(valid.shape, dtype=torch.long, device=valid.device)
    tokens[valid] = MASK_TOKEN
    uncertainty = torch.ones(valid.shape, dtype=torch.float32, device=valid.device)
    with torch.inference_mode():
        for iteration in range(sampling_steps):
            remaining = (tokens == MASK_TOKEN) & valid
            fractions = remaining.sum(dim=(1, 2)).float() / valid.sum(dim=(1, 2)).clamp_min(1)
            logits = model({**conditions, "tokens": tokens, "mask_fraction": fractions[:, None]})
            probability = torch.softmax(logits.float(), dim=1)
            confidence, proposal = probability.max(dim=1)
            for index in range(tokens.shape[0]):
                candidates = torch.nonzero(remaining[index].flatten(), as_tuple=False).flatten()
                if not candidates.numel():
                    continue
                total = int(valid[index].sum())
                target_revealed = math.ceil(total * (iteration + 1) / sampling_steps)
                reveal = max(1, min(int(candidates.numel()), target_revealed - (total - int(candidates.numel()))))
                scores = confidence[index].flatten().index_select(0, candidates)
                chosen = candidates.index_select(0, torch.argsort(scores, descending=True, stable=True)[:reveal])
                tokens[index].flatten()[chosen] = proposal[index].flatten()[chosen]
                uncertainty[index].flatten()[chosen] = 1.0 - confidence[index].flatten()[chosen]
    if bool((tokens[valid] == MASK_TOKEN).any()):
        raise RuntimeError("Prior-v2 sampler failed to reveal every valid token.")
    return {"tokens": tokens, "uncertainty": uncertainty}
