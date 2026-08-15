from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .contract import (
    CELL_STATE_FEATURES,
    EVENT_FEATURES,
    FLUID_SLOTS,
    FLUID_STATE_FEATURES,
    MAX_CELLS,
    SUMMARY_FEATURES,
    CellularPhysiologyTransformerConfig,
)


def _phase_features(phase: Tensor, width: int) -> Tensor:
    if phase.ndim != 1 or width < 16 or not bool(torch.isfinite(phase).all()):
        raise ValueError("cellular physiology phase input drifted")
    quarter = width // 4
    frequency = torch.arange(1, quarter + 1, dtype=torch.float32, device=phase.device)[None]
    angle = phase.float()[:, None] * math.tau * frequency
    encoded = torch.cat((angle.sin(), angle.cos(), (angle * 0.5).sin(), (angle * 0.5).cos()), dim=1)
    return F.pad(encoded, (0, width - encoded.shape[1]))


class PhysiologyConditionEncoder(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.family = nn.Embedding(5, width)
        self.morphotype = nn.Embedding(20, width)
        self.intervention = nn.Embedding(9, width)
        self.phase = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.events = nn.Sequential(nn.Linear(EVENT_FEATURES, width), nn.SiLU(), nn.Linear(width, width))
        self.summary = nn.Sequential(nn.Linear(SUMMARY_FEATURES, width), nn.SiLU(), nn.Linear(width, width))
        self.out = nn.Sequential(nn.LayerNorm(width), nn.SiLU(), nn.Linear(width, width))

    def forward(
        self,
        family: Tensor,
        morphotype: Tensor,
        intervention: Tensor,
        phase: Tensor,
        events: Tensor,
        summary: Tensor,
    ) -> Tensor:
        if (
            family.ndim != 1
            or family.shape != morphotype.shape
            or family.shape != intervention.shape
            or family.shape != phase.shape
            or events.shape != (family.shape[0], EVENT_FEATURES)
            or summary.shape != (family.shape[0], SUMMARY_FEATURES)
            or bool((family < 0).any()) or bool((family >= 5).any())
            or bool((morphotype < 0).any()) or bool((morphotype >= 20).any())
            or bool((intervention < 0).any()) or bool((intervention >= 9).any())
            or not bool(torch.isfinite(events).all())
            or not bool(torch.isfinite(summary).all())
        ):
            raise ValueError("cellular physiology condition contract drifted")
        value = (
            self.family(family)
            + self.morphotype(morphotype)
            + self.intervention(intervention)
            + self.phase(_phase_features(phase, self.width))
            + self.events(events.float())
            + self.summary(summary.float())
        )
        return self.out(value)


class CellularPhysiologyBlock(nn.Module):
    def __init__(self, config: CellularPhysiologyTransformerConfig) -> None:
        super().__init__()
        width = config.width
        self.norm_attention = nn.LayerNorm(width)
        self.norm_graph = nn.LayerNorm(width)
        self.norm_ff = nn.LayerNorm(width)
        self.attention_film = nn.Linear(config.condition_width, width * 2)
        self.ff_film = nn.Linear(config.condition_width, width * 2)
        self.attention = nn.MultiheadAttention(width, config.heads, dropout=config.dropout, batch_first=True)
        self.graph = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        hidden = width * config.feedforward_multiplier
        self.feedforward = nn.Sequential(
            nn.Linear(width, hidden), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(hidden, width)
        )
        self.dropout = nn.Dropout(config.dropout)

    @staticmethod
    def _conditioned(norm: nn.LayerNorm, film: nn.Linear, value: Tensor, condition: Tensor) -> Tensor:
        scale, shift = film(condition).chunk(2, dim=1)
        return norm(value) * (1.0 + scale[:, None]) + shift[:, None]

    def forward(self, value: Tensor, condition: Tensor, mask: Tensor, adjacency: Tensor) -> Tensor:
        attended_input = self._conditioned(self.norm_attention, self.attention_film, value, condition)
        attended, _ = self.attention(
            attended_input, attended_input, attended_input,
            key_padding_mask=~mask, need_weights=False,
        )
        value = value + self.dropout(attended)
        adjacency_float = adjacency.to(value.dtype)
        degree = adjacency_float.sum(dim=2, keepdim=True).clamp_min(1.0)
        neighbors = torch.bmm(adjacency_float, self.norm_graph(value)) / degree
        value = value + self.dropout(self.graph(neighbors))
        fed = self._conditioned(self.norm_ff, self.ff_film, value, condition)
        value = value + self.dropout(self.feedforward(fed))
        return value * mask[:, :, None].to(value.dtype)


class FluidResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width * 3), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(width * 3, width),
        )

    def forward(self, value: Tensor) -> Tensor:
        return value + self.net(value)


class CellularPhysiologyTransformer(nn.Module):
    """Graph transformer for cells plus recurrent point-set puddle dynamics."""

    def __init__(self, config: CellularPhysiologyTransformerConfig | None = None) -> None:
        super().__init__()
        self.config = config or CellularPhysiologyTransformerConfig()
        self.condition = PhysiologyConditionEncoder(self.config.condition_width)
        self.static_projection = nn.Linear(self.config.static_features, self.config.width)
        self.state_projection = nn.Linear(self.config.cell_state_features, self.config.width)
        self.condition_projection = nn.Linear(self.config.condition_width, self.config.width)
        self.blocks = nn.ModuleList(CellularPhysiologyBlock(self.config) for _ in range(self.config.depth))
        self.cell_out = nn.Sequential(
            nn.LayerNorm(self.config.width), nn.SiLU(), nn.Linear(self.config.width, CELL_STATE_FEATURES)
        )
        self.summary_out = nn.Sequential(
            nn.LayerNorm(self.config.width + self.config.condition_width),
            nn.Linear(self.config.width + self.config.condition_width, self.config.width),
            nn.SiLU(), nn.Linear(self.config.width, SUMMARY_FEATURES), nn.Sigmoid(),
        )
        self.fluid_query = nn.Embedding(FLUID_SLOTS, self.config.fluid_width)
        self.fluid_state = nn.Linear(FLUID_STATE_FEATURES, self.config.fluid_width)
        self.fluid_global = nn.Linear(self.config.width + self.config.condition_width, self.config.fluid_width)
        self.fluid_blocks = nn.ModuleList(
            FluidResidualBlock(self.config.fluid_width, self.config.dropout)
            for _ in range(self.config.fluid_depth)
        )
        self.fluid_out = nn.Sequential(
            nn.LayerNorm(self.config.fluid_width), nn.SiLU(), nn.Linear(self.config.fluid_width, FLUID_STATE_FEATURES)
        )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(
        self,
        static: Tensor,
        cell_state: Tensor,
        summary_state: Tensor,
        fluid_state: Tensor,
        mask: Tensor,
        adjacency: Tensor,
        family: Tensor,
        morphotype: Tensor,
        intervention: Tensor,
        phase: Tensor,
        events: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch = static.shape[0]
        if (
            static.shape != (batch, MAX_CELLS, self.config.static_features)
            or cell_state.shape != (batch, MAX_CELLS, CELL_STATE_FEATURES)
            or summary_state.shape != (batch, SUMMARY_FEATURES)
            or fluid_state.shape != (batch, FLUID_SLOTS, FLUID_STATE_FEATURES)
            or mask.shape != (batch, MAX_CELLS) or mask.dtype is not torch.bool
            or adjacency.shape != (batch, MAX_CELLS, MAX_CELLS) or adjacency.dtype is not torch.bool
            or bool((adjacency & ~(mask[:, :, None] & mask[:, None])).any())
            or not bool(mask.any(dim=1).all())
            or not all(bool(torch.isfinite(value).all()) for value in (static, cell_state, summary_state, fluid_state))
        ):
            raise ValueError("cellular physiology model input contract drifted")
        condition = self.condition(family, morphotype, intervention, phase, events, summary_state)
        value = (
            self.static_projection(static.float())
            + self.state_projection(cell_state.float())
            + self.condition_projection(condition)[:, None]
        ) * mask[:, :, None].to(static.dtype)
        for block in self.blocks:
            value = block(value, condition, mask, adjacency)
        raw_cell = self.cell_out(value)
        cell = torch.cat(
            (raw_cell[:, :, :2].tanh(), raw_cell[:, :, 2:].sigmoid()), dim=2
        ) * mask[:, :, None].to(raw_cell.dtype)
        pooled = (value * mask[:, :, None]).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)
        global_value = torch.cat((pooled, condition), dim=1)
        summary = self.summary_out(global_value)
        fluid = self.fluid_state(fluid_state.float())
        fluid = fluid + self.fluid_query.weight[None] + self.fluid_global(global_value)[:, None]
        for block in self.fluid_blocks:
            fluid = block(fluid)
        raw_fluid = self.fluid_out(fluid)
        fluid_output = torch.cat(
            (raw_fluid[:, :, :4].tanh(), raw_fluid[:, :, 4:].sigmoid()), dim=2
        )
        return cell, summary, fluid_output


def _masked_mean(value: Tensor, mask: Tensor) -> Tensor:
    expanded = mask[:, :, None].to(value.dtype)
    return (value * expanded).sum() / expanded.sum().clamp_min(1.0) / value.shape[-1]


def cellular_physiology_loss(
    predicted: tuple[Tensor, Tensor, Tensor],
    cell_target: Tensor,
    summary_target: Tensor,
    fluid_target: Tensor,
    mask: Tensor,
    adjacency: Tensor,
) -> tuple[Tensor, dict[str, Tensor]]:
    cell, summary, fluid = predicted
    if cell.shape != cell_target.shape or summary.shape != summary_target.shape or fluid.shape != fluid_target.shape:
        raise ValueError("cellular physiology loss tensor shape drifted")
    position = _masked_mean(F.smooth_l1_loss(cell[:, :, :2], cell_target[:, :, :2], reduction="none"), mask)
    health = _masked_mean(F.smooth_l1_loss(cell[:, :, 2:3], cell_target[:, :, 2:3], reduction="none"), mask)
    alive = _masked_mean(F.binary_cross_entropy(cell[:, :, 3:4], cell_target[:, :, 3:4], reduction="none"), mask)
    summary_loss = F.smooth_l1_loss(summary, summary_target)
    target_present = fluid_target[:, :, 6:7]
    presence = F.binary_cross_entropy(fluid[:, :, 6:7], target_present)
    fluid_values = (
        F.smooth_l1_loss(fluid[:, :, :6], fluid_target[:, :, :6], reduction="none")
        * target_present
    ).sum() / target_present.sum().clamp_min(1.0) / 6.0
    adjacency_float = adjacency.to(cell.dtype)
    degree = adjacency_float.sum(dim=2, keepdim=True).clamp_min(1.0)
    predicted_neighbors = torch.bmm(adjacency_float, cell[:, :, 2:3]) / degree
    target_neighbors = torch.bmm(adjacency_float, cell_target[:, :, 2:3]) / degree
    graph = _masked_mean(
        F.smooth_l1_loss(cell[:, :, 2:3] - predicted_neighbors, cell_target[:, :, 2:3] - target_neighbors, reduction="none"),
        mask,
    )
    outside = cell[~mask].abs().max()
    total = (
        position + health * 1.2 + alive * 0.45 + summary_loss * 0.8
        + presence * 0.35 + fluid_values * 0.5 + graph * 0.25 + outside
    )
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("cellular physiology loss became non-finite")
    return total, {
        "loss": total.detach(), "position": position.detach(), "health": health.detach(),
        "alive": alive.detach(), "summary": summary_loss.detach(), "fluid_presence": presence.detach(),
        "fluid_values": fluid_values.detach(), "graph": graph.detach(), "outside": outside.detach(),
    }
