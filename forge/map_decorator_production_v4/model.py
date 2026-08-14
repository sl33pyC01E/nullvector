from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from ..map_decorator_ml.contract import FEATURE_CHANNEL_COUNT, ModelConfig
from ..map_decorator_ml.model import CategoricalRefinementUNet, HeadLogits
from ..map_decorator_production_v2.model import OBJECT_HEADS, compose_object_logits
from .contract import DECAL_PROPOSAL_CHANNELS, PROP_PROPOSAL_CHANNELS, ProposalLocatorConfig


PROPOSAL_CHANNELS = {"decal": DECAL_PROPOSAL_CHANNELS, "prop": PROP_PROPOSAL_CHANNELS}


class ProposalResidualTower(nn.Module):
    def __init__(self, proposal_channels: int, config: ProposalLocatorConfig) -> None:
        super().__init__()
        channels = config.locator_channels
        self.stem = nn.Conv2d(FEATURE_CHANNEL_COUNT + 3 + proposal_channels, channels, 3, padding=1)
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
        self.presence_residual = nn.Conv2d(channels, 1, 1)
        self.count_residual = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, config.count_hidden_channels),
            nn.SiLU(),
            nn.Linear(config.count_hidden_channels, 1),
        )
        with torch.no_grad():
            self.presence_residual.weight.zero_()
            self.presence_residual.bias.zero_()
            self.count_residual[-1].weight.zero_()
            self.count_residual[-1].bias.zero_()

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.stem(value)
        for block in self.blocks:
            hidden = hidden + block(hidden)
        return self.presence_residual(hidden).squeeze(1), self.count_residual(hidden).squeeze(1)


@dataclass(slots=True)
class ProposalLocatorOutputV4:
    categorical: HeadLogits
    presence_logits: dict[str, torch.Tensor]
    type_logits: dict[str, torch.Tensor]
    log1p_counts: dict[str, torch.Tensor]
    proposals: dict[str, torch.Tensor]
    maximum_objects_per_head: int

    def as_head_logits(self) -> HeadLogits:
        return self.categorical

    @property
    def variant(self) -> torch.Tensor:
        return self.categorical.variant

    @property
    def emission(self) -> torch.Tensor:
        return self.categorical.emission


class ProposalConditionedDecoratorV4(nn.Module):
    """Neural residual decorator over exact public-entropy object proposals."""

    def __init__(
        self,
        core_config: ModelConfig = ModelConfig(),
        locator_config: ProposalLocatorConfig = ProposalLocatorConfig(),
    ) -> None:
        super().__init__()
        self.core_config = core_config
        self.locator_config = locator_config
        self.core = CategoricalRefinementUNet(core_config)
        self.locators = nn.ModuleDict(
            {
                name: ProposalResidualTower(PROPOSAL_CHANNELS[name], locator_config)
                for name in OBJECT_HEADS
            }
        )

    def forward(
        self,
        features: torch.Tensor,
        labels: Mapping[str, torch.Tensor],
        masked: Mapping[str, torch.Tensor],
        theme_index: torch.Tensor,
        global_conditions: torch.Tensor,
        refinement_level: torch.Tensor,
        proposals: Mapping[str, torch.Tensor],
    ) -> ProposalLocatorOutputV4:
        raw = self.core(features, labels, masked, theme_index, global_conditions, refinement_level)
        expected_shape = features.shape[0], features.shape[2], features.shape[3]
        normalized: dict[str, torch.Tensor] = {}
        for name in OBJECT_HEADS:
            proposal = proposals.get(name)
            expected = (expected_shape[0], PROPOSAL_CHANNELS[name], expected_shape[1], expected_shape[2])
            if not isinstance(proposal, torch.Tensor) or proposal.dtype != torch.bool or tuple(proposal.shape) != expected:
                raise TypeError(f"V4 {name} proposals must be boolean with shape {expected}.")
            if proposal.device != features.device:
                raise ValueError("V4 proposals and features must share a device.")
            normalized[name] = proposal

        presence: dict[str, torch.Tensor] = {}
        types: dict[str, torch.Tensor] = {}
        counts: dict[str, torch.Tensor] = {}
        composed = {"variant": raw.variant, "emission": raw.emission}
        for name in OBJECT_HEADS:
            raw_logits = getattr(raw, name)
            proposal_float = normalized[name].to(features.dtype)
            presence_residual, count_residual = self.locators[name](
                torch.cat((features, raw_logits, proposal_float), dim=1)
            )
            candidate = normalized[name].any(dim=1)
            prior = torch.where(
                candidate,
                torch.full_like(presence_residual, self.locator_config.candidate_logit_prior),
                torch.full_like(presence_residual, self.locator_config.noncandidate_logit_prior),
            )
            presence[name] = prior + presence_residual
            type_prior = torch.where(
                normalized[name],
                torch.full_like(raw_logits[:, 1:], self.locator_config.proposal_type_logit_prior),
                torch.full_like(raw_logits[:, 1:], -self.locator_config.proposal_type_logit_prior),
            )
            types[name] = raw_logits[:, 1:] + type_prior
            candidate_count = candidate.sum(dim=(1, 2)).to(torch.float32)
            counts[name] = torch.clamp(
                torch.log1p(candidate_count)
                + self.locator_config.count_residual_scale * torch.tanh(count_residual.to(torch.float32)),
                min=0.0,
            )
            composed[name] = compose_object_logits(presence[name], types[name])
        tensors = (*presence.values(), *types.values(), *counts.values())
        if any(not bool(torch.isfinite(value).all()) for value in tensors):
            raise FloatingPointError("V4 proposal-conditioned locator produced non-finite output.")
        return ProposalLocatorOutputV4(
            categorical=HeadLogits(**{name: composed[name] for name in ("variant", "decal", "prop", "emission")}),
            presence_logits=presence,
            type_logits=types,
            log1p_counts=counts,
            proposals=normalized,
            maximum_objects_per_head=self.locator_config.maximum_objects_per_head,
        )
