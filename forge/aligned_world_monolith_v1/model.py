from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ..whole_viewport_latent_v1.contract import ModelConfig as RendererConfig
from ..whole_viewport_latent_v1.model import WholeViewportLatentModel
from .contract import ModelConfig


class ConditionedResidual(nn.Module):
    def __init__(self, width: int, dilation: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(16, width)
        self.conv1 = nn.Conv2d(width, width, 3, padding=dilation, dilation=dilation)
        self.norm2 = nn.GroupNorm(16, width)
        self.conv2 = nn.Conv2d(width, width, 3, padding=1)
        self.condition = nn.Linear(width, width * 2)

    def forward(self, value: Tensor, condition: Tensor) -> Tensor:
        scale, bias = self.condition(condition).chunk(2, 1)
        hidden = self.conv1(F.silu(self.norm1(value)))
        hidden = self.norm2(hidden) * (1 + scale[:, :, None, None]) + bias[:, :, None, None]
        return value + self.conv2(F.silu(hidden))


@dataclass
class WorldStep:
    spatial: Tensor
    organisms: Tensor
    organism_probability: Tensor
    state: Tensor
    actor_state: Tensor
    actor_field: Tensor
    visibility: Tensor
    memory: Tensor
    timeline: Tensor
    timeline_event: Tensor
    counterfactual: Tensor
    latent: Tensor | None = None


class AlignedWorldMonolith(nn.Module):
    """One causal graph: physical state transition, then full-viewport latent rendering."""

    def __init__(self, config: ModelConfig = ModelConfig(), renderer_config: RendererConfig = RendererConfig()):
        super().__init__()
        self.config, self.renderer_config = config, renderer_config
        width = config.width
        self.renderer = WholeViewportLatentModel(renderer_config)
        self.scene = nn.Conv2d(68 + 8 + 1 + 1 + 48, width, 3, padding=1)
        self.organism_encoder = nn.Sequential(nn.LayerNorm(164), nn.Linear(164, width), nn.SiLU(), nn.Linear(width, width))
        self.organism_mix = nn.Conv2d(width, width, 3, padding=1)
        self.action = nn.Embedding(22, width)
        self.condition = nn.Sequential(nn.Linear(64 + 128 + 4, width * 2), nn.SiLU(), nn.Linear(width * 2, width))
        self.blocks = nn.ModuleList(ConditionedResidual(width, (1, 2, 3, 1, 2, 1)[index % 6]) for index in range(config.blocks))
        self.spatial_out = nn.Conv2d(width, 68, 3, padding=1)
        self.actor_field_out = nn.Conv2d(width, 8, 3, padding=1)
        self.visibility_out = nn.Conv2d(width, 1, 3, padding=1)
        self.memory_out = nn.Conv2d(width, 1, 3, padding=1)
        self.slot_embedding = nn.Parameter(torch.randn(1, config.organism_slots, width) * .015)
        self.organism_update = nn.Sequential(nn.Linear(width * 3, config.organism_width), nn.SiLU(), nn.Linear(config.organism_width, 165))
        self.global_update = nn.Sequential(nn.Linear(width * 2, width * 2), nn.SiLU(), nn.Linear(width * 2, 64 + 128 + 3 + 10 + 20))
        for head in (self.spatial_out, self.actor_field_out, self.visibility_out, self.memory_out):
            nn.init.zeros_(head.weight); nn.init.zeros_(head.bias)
        nn.init.zeros_(self.organism_update[-1].weight); nn.init.zeros_(self.organism_update[-1].bias)
        nn.init.zeros_(self.global_update[-1].weight); nn.init.zeros_(self.global_update[-1].bias)

    def load_renderer_parent(self, state: dict[str, Tensor]) -> None:
        self.renderer.load_state_dict(state, strict=True)

    @staticmethod
    def _bounded(current: Tensor, delta: Tensor, low: float = -1, high: float = 1) -> Tensor:
        return (current + torch.tanh(delta)).clamp(low, high)

    def transition(self, latent: Tensor, spatial: Tensor, organisms: Tensor, organism_mask: Tensor,
                   state: Tensor, actor_state: Tensor, actor_field: Tensor, visibility: Tensor,
                   memory: Tensor, control: Tensor, action: Tensor, timeline: Tensor,
                   counterfactual: Tensor) -> WorldStep:
        batch = len(spatial)
        if spatial.shape[1:] != (68, 32, 32) or organisms.shape[1:] != (64, 164) or latent.shape[1:] != (48, 32, 32):
            raise ValueError("aligned world input contract drifted")
        organism_tokens = self.organism_encoder(organisms)
        splat = WholeViewportLatentModel._splat(organism_tokens, organisms[:, :, :2], organism_mask)
        hidden = self.scene(torch.cat((spatial, actor_field, visibility, memory, latent), 1)) + self.organism_mix(splat)
        condition = self.condition(torch.cat((state, actor_state, control), 1)) + self.action(action.reshape(batch))
        for block in self.blocks:
            hidden = block(hidden, condition)
        positions = (organisms[:, :, :2].clamp(-.5, .5) * 2).reshape(batch, 64, 1, 2)
        sampled = F.grid_sample(hidden, positions, mode="bilinear", padding_mode="border", align_corners=False).squeeze(-1).transpose(1, 2)
        organism_result = self.organism_update(torch.cat((organism_tokens, sampled, self.slot_embedding.expand(batch, -1, -1)), -1))
        organism_delta, mask_delta = organism_result[:, :, :164], organism_result[:, :, 164]
        prior_logit = torch.where(organism_mask.bool(), mask_delta.new_full(mask_delta.shape, 6), mask_delta.new_full(mask_delta.shape, -6))
        probability = torch.sigmoid(prior_logit + mask_delta)
        pooled = hidden.mean((2, 3))
        global_result = self.global_update(torch.cat((pooled, condition), 1))
        cursor = 0
        state_delta, cursor = global_result[:, cursor:cursor + 64], cursor + 64
        actor_delta, cursor = global_result[:, cursor:cursor + 128], cursor + 128
        timeline_delta, cursor = global_result[:, cursor:cursor + 3], cursor + 3
        event_logits, cursor = global_result[:, cursor:cursor + 10], cursor + 10
        counter_delta = global_result[:, cursor:].reshape(batch, 5, 4)
        return WorldStep(
            self._bounded(spatial, self.spatial_out(hidden)),
            self._bounded(organisms, organism_delta, -2, 2), probability,
            self._bounded(state, state_delta, -1, 2), self._bounded(actor_state, actor_delta, -2, 2),
            self._bounded(actor_field, self.actor_field_out(hidden)),
            self._bounded(visibility, self.visibility_out(hidden), 0, 1),
            self._bounded(memory, self.memory_out(hidden), 0, 1),
            self._bounded(timeline, timeline_delta, -1, 2), event_logits,
            self._bounded(counterfactual, counter_delta, -2, 2), None,
        )

    def render(self, previous_latent: Tensor, step: WorldStep, control: Tensor, action: Tensor) -> Tensor:
        return self.renderer(previous_latent, step.spatial, step.organisms, step.organism_probability,
                             step.state, step.actor_state, step.actor_field, step.visibility, step.memory,
                             control, action)

    def forward(self, latent: Tensor, spatial: Tensor, organisms: Tensor, organism_mask: Tensor,
                state: Tensor, actor_state: Tensor, actor_field: Tensor, visibility: Tensor,
                memory: Tensor, control: Tensor, action: Tensor, timeline: Tensor,
                counterfactual: Tensor) -> WorldStep:
        step = self.transition(latent, spatial, organisms, organism_mask, state, actor_state, actor_field,
                               visibility, memory, control, action, timeline, counterfactual)
        step.latent = self.render(latent, step, control, action)
        return step
