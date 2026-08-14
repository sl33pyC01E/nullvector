from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from ..map_decorator_ml.contract import FEATURE_CHANNEL_COUNT, HEAD_NAMES
from ..map_decorator_ml.model import CategoricalRefinementUNet, HeadLogits
from ..map_decorator_production_v2.model import OBJECT_HEADS, compose_object_logits
from .contract import LocatorModelConfig


def _inverse_softplus(value: float) -> float:
    if value <= 0:
        return -20.0
    return math.log(math.expm1(value))


class SpatialLocatorTower(nn.Module):
    def __init__(self, input_channels: int, config: LocatorModelConfig) -> None:
        super().__init__()
        channels = config.locator_channels
        self.stem = nn.Conv2d(input_channels, channels, 3, padding=1)
        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.GroupNorm(1, channels),
                nn.SiLU(),
                nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
                nn.SiLU(),
                nn.Conv2d(channels, channels, 1),
            )
            for _ in range(config.locator_blocks)
        )
        self.presence = nn.Conv2d(channels, 1, 1)
        self.count = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, config.count_hidden_channels),
            nn.SiLU(),
            nn.Linear(config.count_hidden_channels, 1),
        )
        with torch.no_grad():
            self.presence.weight.zero_()
            self.presence.bias.fill_(config.presence_bias_init)
            self.count[-1].weight.zero_()
            self.count[-1].bias.fill_(_inverse_softplus(math.log1p(config.count_prior)))

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.stem(value)
        for block in self.blocks:
            hidden = hidden + block(hidden)
        presence = self.presence(hidden).squeeze(1)
        log1p_count = F.softplus(self.count(hidden).squeeze(1))
        return presence, log1p_count


@dataclass(slots=True)
class SparseLocatorOutput:
    categorical: HeadLogits
    presence_logits: dict[str, torch.Tensor]
    type_logits: dict[str, torch.Tensor]
    log1p_counts: dict[str, torch.Tensor]
    maximum_objects_per_head: int

    def as_head_logits(self) -> HeadLogits:
        return self.categorical

    @property
    def variant(self) -> torch.Tensor:
        return self.categorical.variant

    @property
    def emission(self) -> torch.Tensor:
        return self.categorical.emission


class SparseLocatorDecoratorV3(nn.Module):
    """Categorical decorator with decoupled spatial ranking and object counts."""

    def __init__(self, config: LocatorModelConfig = LocatorModelConfig()) -> None:
        super().__init__()
        self.config = config
        self.core = CategoricalRefinementUNet(config.core_config())
        locator_input_channels = FEATURE_CHANNEL_COUNT + 3
        self.locators = nn.ModuleDict(
            {name: SpatialLocatorTower(locator_input_channels, config) for name in OBJECT_HEADS}
        )

    def forward(
        self,
        features: torch.Tensor,
        labels: Mapping[str, torch.Tensor],
        masked: Mapping[str, torch.Tensor],
        theme_index: torch.Tensor,
        global_conditions: torch.Tensor,
        refinement_level: torch.Tensor,
    ) -> SparseLocatorOutput:
        raw = self.core(features, labels, masked, theme_index, global_conditions, refinement_level)
        presence: dict[str, torch.Tensor] = {}
        types: dict[str, torch.Tensor] = {}
        counts: dict[str, torch.Tensor] = {}
        composed = {"variant": raw.variant, "emission": raw.emission}
        for name in OBJECT_HEADS:
            raw_logits = getattr(raw, name)
            locator_input = torch.cat((features, raw_logits), dim=1)
            presence[name], counts[name] = self.locators[name](locator_input)
            types[name] = raw_logits[:, 1:]
            composed[name] = compose_object_logits(presence[name], types[name])
        if any(not bool(torch.isfinite(value).all()) for value in (*presence.values(), *counts.values())):
            raise FloatingPointError("Sparse locator produced a non-finite output.")
        return SparseLocatorOutput(
            categorical=HeadLogits(**{name: composed[name] for name in HEAD_NAMES}),
            presence_logits=presence,
            type_logits=types,
            log1p_counts=counts,
            maximum_objects_per_head=self.config.maximum_objects_per_head,
        )
