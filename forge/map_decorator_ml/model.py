from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from ..maps.model import THEMES
from .contract import (
    FEATURE_CHANNEL_COUNT,
    GLOBAL_CONDITION_DIM,
    HEAD_CLASS_COUNTS,
    HEAD_NAMES,
    ModelConfig,
)


@dataclass(slots=True)
class HeadLogits:
    variant: torch.Tensor
    decal: torch.Tensor
    prop: torch.Tensor
    emission: torch.Tensor

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {name: getattr(self, name) for name in HEAD_NAMES}


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConditionedDepthwiseBlock(nn.Module):
    def __init__(self, channels: int, condition_channels: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.depthwise = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.pointwise = nn.Conv2d(channels, channels, 1)
        self.film = nn.Linear(condition_channels, channels * 2)

    def forward(self, value: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        scale, shift = self.film(condition).chunk(2, dim=1)
        hidden = self.norm(value)
        hidden = hidden * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        hidden = F.silu(hidden)
        hidden = self.depthwise(hidden)
        hidden = F.silu(hidden)
        return value + self.pointwise(hidden)


class BlockStack(nn.Module):
    def __init__(self, channels: int, condition_channels: int, count: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            ConditionedDepthwiseBlock(channels, condition_channels) for _ in range(count)
        )

    def forward(self, value: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            value = block(value, condition)
        return value


def _validate_spatial(tensor: torch.Tensor, name: str, channels: int | None = None) -> None:
    if tensor.ndim != 4:
        raise ValueError(f"{name} must have shape [B,C,H,W].")
    if channels is not None and tensor.shape[1] != channels:
        raise ValueError(f"{name} must have {channels} channels.")
    height, width = tensor.shape[-2:]
    if not (32 <= height <= 256 and 32 <= width <= 256):
        raise ValueError(f"{name} spatial dimensions must each be in [32, 256].")


def pad_right_bottom(tensor: torch.Tensor, multiple: int) -> tuple[torch.Tensor, tuple[int, int]]:
    """Pad without shifting source cells; returned shape records the exact crop."""
    height, width = tensor.shape[-2:]
    pad_h = (-height) % multiple
    pad_w = (-width) % multiple
    if pad_h or pad_w:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
    return tensor, (height, width)


def crop_exact(tensor: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    height, width = shape
    if tensor.shape[-2] < height or tensor.shape[-1] < width:
        raise ValueError("Cannot crop a tensor smaller than the requested source shape.")
    return tensor[..., :height, :width]


class CategoricalRefinementUNet(nn.Module):
    """Compact, grid-aligned categorical U-Net with FiLM global conditioning."""

    def __init__(self, config: ModelConfig = ModelConfig()) -> None:
        super().__init__()
        self.config = config
        state_channels = sum(HEAD_CLASS_COUNTS.values()) + len(HEAD_NAMES) + 1
        input_channels = FEATURE_CHANNEL_COUNT + state_channels
        base = config.base_channels
        condition_channels = config.condition_channels
        blocks = config.residual_blocks_per_scale

        self.theme_embedding = nn.Embedding(len(THEMES), condition_channels)
        self.global_condition = nn.Sequential(
            nn.Linear(GLOBAL_CONDITION_DIM, condition_channels),
            nn.SiLU(),
            nn.Linear(condition_channels, condition_channels),
        )
        self.level_condition = nn.Sequential(
            nn.Linear(1, condition_channels), nn.SiLU(), nn.Linear(condition_channels, condition_channels)
        )
        self.stem = nn.Conv2d(input_channels, base, 3, padding=1)
        self.enc0 = BlockStack(base, condition_channels, blocks)
        self.down1 = nn.Conv2d(base, base * 2, 3, stride=2, padding=1)
        self.enc1 = BlockStack(base * 2, condition_channels, blocks)
        self.down2 = nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1)
        self.middle = BlockStack(base * 4, condition_channels, blocks + 1)
        self.up1_projection = nn.Conv2d(base * 4, base * 2, 1)
        self.up1_merge = nn.Conv2d(base * 4, base * 2, 1)
        self.dec1 = BlockStack(base * 2, condition_channels, blocks)
        self.up0_projection = nn.Conv2d(base * 2, base, 1)
        self.up0_merge = nn.Conv2d(base * 2, base, 1)
        self.dec0 = BlockStack(base, condition_channels, blocks)
        self.output_norm = nn.GroupNorm(_group_count(base), base)
        self.heads = nn.ModuleDict(
            {name: nn.Conv2d(base, classes, 1) for name, classes in HEAD_CLASS_COUNTS.items()}
        )

    def _state_tensor(
        self,
        labels: Mapping[str, torch.Tensor],
        masked: Mapping[str, torch.Tensor],
        refinement_level: torch.Tensor,
        shape: tuple[int, int, int],
    ) -> torch.Tensor:
        batch, height, width = shape
        channels: list[torch.Tensor] = []
        for name, classes in HEAD_CLASS_COUNTS.items():
            if name not in labels or name not in masked:
                raise ValueError(f"Missing state field {name!r}.")
            label = labels[name]
            mask = masked[name]
            if label.shape != (batch, height, width) or label.dtype != torch.long:
                raise TypeError(f"labels[{name!r}] must be int64 [B,H,W].")
            if mask.shape != label.shape or mask.dtype != torch.bool:
                raise TypeError(f"masked[{name!r}] must be bool [B,H,W].")
            if bool(((label < 0) | (label >= classes)).any()):
                raise ValueError(f"labels[{name!r}] contains an out-of-domain class.")
            one_hot = F.one_hot(label, num_classes=classes).permute(0, 3, 1, 2)
            channels.append(one_hot.to(dtype=torch.float32) * (~mask[:, None]).to(torch.float32))
        channels.extend(masked[name][:, None].to(torch.float32) for name in HEAD_NAMES)
        if refinement_level.shape != (batch,):
            raise ValueError("refinement_level must have shape [B].")
        if bool(((refinement_level < 0) | (refinement_level > 1)).any()):
            raise ValueError("refinement_level must stay in [0, 1].")
        channels.append(refinement_level[:, None, None, None].expand(batch, 1, height, width))
        return torch.cat(channels, dim=1)

    def forward(
        self,
        features: torch.Tensor,
        labels: Mapping[str, torch.Tensor],
        masked: Mapping[str, torch.Tensor],
        theme_index: torch.Tensor,
        global_conditions: torch.Tensor,
        refinement_level: torch.Tensor,
    ) -> HeadLogits:
        _validate_spatial(features, "features", FEATURE_CHANNEL_COUNT)
        if features.dtype != torch.float32 or not bool(torch.isfinite(features).all()):
            raise TypeError("features must be a finite float32 tensor.")
        batch, _, height, width = features.shape
        if theme_index.shape != (batch,) or theme_index.dtype != torch.long:
            raise TypeError("theme_index must be int64 [B].")
        if bool(((theme_index < 0) | (theme_index >= len(THEMES))).any()):
            raise ValueError("theme_index contains an unknown theme.")
        if global_conditions.shape != (batch, GLOBAL_CONDITION_DIM):
            raise ValueError(f"global_conditions must have shape [B,{GLOBAL_CONDITION_DIM}].")
        if global_conditions.dtype != torch.float32 or not bool(torch.isfinite(global_conditions).all()):
            raise TypeError("global_conditions must be a finite float32 tensor.")
        if refinement_level.dtype != torch.float32:
            raise TypeError("refinement_level must use dtype float32.")
        state = self._state_tensor(labels, masked, refinement_level, (batch, height, width))
        model_input = torch.cat((features, state), dim=1)
        model_input, original_shape = pad_right_bottom(model_input, self.config.padding_multiple)
        condition = (
            self.theme_embedding(theme_index)
            + self.global_condition(global_conditions.to(torch.float32))
            + self.level_condition(refinement_level[:, None].to(torch.float32))
        )

        skip0 = self.enc0(self.stem(model_input), condition)
        skip1 = self.enc1(self.down1(skip0), condition)
        hidden = self.middle(self.down2(skip1), condition)
        hidden = F.interpolate(hidden, size=skip1.shape[-2:], mode="nearest")
        hidden = self.up1_projection(hidden)
        hidden = self.dec1(self.up1_merge(torch.cat((hidden, skip1), dim=1)), condition)
        hidden = F.interpolate(hidden, size=skip0.shape[-2:], mode="nearest")
        hidden = self.up0_projection(hidden)
        hidden = self.dec0(self.up0_merge(torch.cat((hidden, skip0), dim=1)), condition)
        hidden = F.silu(self.output_norm(hidden))
        outputs = {
            name: crop_exact(head(hidden), original_shape) for name, head in self.heads.items()
        }
        return HeadLogits(**outputs)
