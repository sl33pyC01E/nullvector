from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .contract import (
    CONTROL_FEATURES,
    MAX_CELLS,
    CellularMotionTransformerConfig,
)


def _phase_features(phase: Tensor, width: int) -> Tensor:
    if phase.ndim != 1 or width < 16 or not bool(torch.isfinite(phase).all()):
        raise ValueError("cellular motion phase input drifted")
    quarter = width // 4
    frequency = torch.arange(1, quarter + 1, dtype=torch.float32, device=phase.device)[None]
    angle = phase.float()[:, None] * math.tau * frequency
    encoded = torch.cat((angle.sin(), angle.cos(), (angle * 0.5).sin(), (angle * 0.5).cos()), dim=1)
    return F.pad(encoded, (0, width - encoded.shape[1]))


class ConditionEncoder(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.family = nn.Embedding(5, width)
        self.morphotype = nn.Embedding(20, width)
        self.motion = nn.Embedding(13, width)
        self.phase = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.controls = nn.Sequential(
            nn.Linear(CONTROL_FEATURES, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.out = nn.Sequential(nn.LayerNorm(width), nn.SiLU(), nn.Linear(width, width))

    def forward(
        self,
        family: Tensor,
        morphotype: Tensor,
        motion: Tensor,
        phase: Tensor,
        controls: Tensor,
    ) -> Tensor:
        if (
            family.ndim != 1
            or family.shape != morphotype.shape
            or family.shape != motion.shape
            or family.shape != phase.shape
            or controls.shape != (family.shape[0], CONTROL_FEATURES)
            or bool((family < 0).any())
            or bool((family >= 5).any())
            or bool((morphotype < 0).any())
            or bool((morphotype >= 20).any())
            or bool((motion < 0).any())
            or bool((motion >= 13).any())
            or not bool(torch.isfinite(controls).all())
        ):
            raise ValueError("cellular motion condition contract drifted")
        value = (
            self.family(family)
            + self.morphotype(morphotype)
            + self.motion(motion)
            + self.phase(_phase_features(phase, self.width))
            + self.controls(controls.float())
        )
        return self.out(value)


class CellularTransformerBlock(nn.Module):
    def __init__(self, config: CellularMotionTransformerConfig) -> None:
        super().__init__()
        width = config.width
        self.norm_attention = nn.LayerNorm(width)
        self.norm_graph = nn.LayerNorm(width)
        self.norm_ff = nn.LayerNorm(width)
        self.attention_film = nn.Linear(config.condition_width, width * 2)
        self.ff_film = nn.Linear(config.condition_width, width * 2)
        self.attention = nn.MultiheadAttention(
            width, config.heads, dropout=config.dropout, batch_first=True
        )
        self.graph = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        hidden = width * config.feedforward_multiplier
        self.feedforward = nn.Sequential(
            nn.Linear(width, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, width),
        )
        self.dropout = nn.Dropout(config.dropout)

    @staticmethod
    def _conditioned(norm: nn.LayerNorm, film: nn.Linear, value: Tensor, condition: Tensor) -> Tensor:
        scale, shift = film(condition).chunk(2, dim=1)
        return norm(value) * (1.0 + scale[:, None, :]) + shift[:, None, :]

    def forward(
        self,
        value: Tensor,
        condition: Tensor,
        mask: Tensor,
        adjacency: Tensor,
    ) -> Tensor:
        attended_input = self._conditioned(
            self.norm_attention, self.attention_film, value, condition
        )
        attended, _ = self.attention(
            attended_input,
            attended_input,
            attended_input,
            key_padding_mask=~mask,
            need_weights=False,
        )
        value = value + self.dropout(attended)
        adjacency_float = adjacency.to(dtype=value.dtype)
        degree = adjacency_float.sum(dim=2, keepdim=True).clamp_min(1.0)
        neighbors = torch.bmm(adjacency_float, self.norm_graph(value)) / degree
        value = value + self.dropout(self.graph(neighbors))
        fed = self._conditioned(self.norm_ff, self.ff_film, value, condition)
        value = value + self.dropout(self.feedforward(fed))
        return value * mask[:, :, None].to(value.dtype)


class CellularMotionTransformer(nn.Module):
    """Recurrent cell-token transformer with explicit local bond aggregation."""

    def __init__(self, config: CellularMotionTransformerConfig | None = None) -> None:
        super().__init__()
        self.config = config or CellularMotionTransformerConfig()
        self.condition = ConditionEncoder(self.config.condition_width)
        self.static_projection = nn.Linear(self.config.static_features, self.config.width)
        self.state_projection = nn.Linear(self.config.state_features, self.config.width)
        self.condition_projection = nn.Linear(self.config.condition_width, self.config.width)
        self.blocks = nn.ModuleList(
            CellularTransformerBlock(self.config) for _ in range(self.config.depth)
        )
        self.out = nn.Sequential(
            nn.LayerNorm(self.config.width),
            nn.SiLU(),
            nn.Linear(self.config.width, self.config.output_features),
            nn.Tanh(),
        )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(
        self,
        static: Tensor,
        state: Tensor,
        mask: Tensor,
        adjacency: Tensor,
        family: Tensor,
        morphotype: Tensor,
        motion: Tensor,
        phase: Tensor,
        controls: Tensor,
    ) -> Tensor:
        batch = static.shape[0]
        if (
            static.shape != (batch, MAX_CELLS, self.config.static_features)
            or state.shape != (batch, MAX_CELLS, self.config.state_features)
            or mask.shape != (batch, MAX_CELLS)
            or mask.dtype is not torch.bool
            or adjacency.shape != (batch, MAX_CELLS, MAX_CELLS)
            or adjacency.dtype is not torch.bool
            or not bool(mask.any(dim=1).all())
            or bool((adjacency & ~(mask[:, :, None] & mask[:, None, :])).any())
            or not bool(torch.isfinite(static).all())
            or not bool(torch.isfinite(state).all())
        ):
            raise ValueError("cellular motion model input contract drifted")
        condition = self.condition(family, morphotype, motion, phase, controls)
        value = (
            self.static_projection(static.float())
            + self.state_projection(state.float())
            + self.condition_projection(condition)[:, None, :]
        )
        value = value * mask[:, :, None].to(value.dtype)
        for block in self.blocks:
            value = block(value, condition, mask, adjacency)
        return self.out(value) * mask[:, :, None].to(value.dtype)


def _masked_mean(value: Tensor, mask: Tensor) -> Tensor:
    expanded = mask[:, :, None].to(value.dtype)
    return (value * expanded).sum() / expanded.sum().clamp_min(1.0) / value.shape[-1]


def cellular_motion_loss(
    predicted: Tensor,
    target: Tensor,
    previous: Tensor,
    mask: Tensor,
    adjacency: Tensor,
) -> tuple[Tensor, dict[str, Tensor]]:
    if predicted.shape != target.shape or predicted.shape != previous.shape:
        raise ValueError("cellular motion loss tensor shape drifted")
    if predicted.shape[-1] != 4 or mask.shape != predicted.shape[:2]:
        raise ValueError("cellular motion loss interface drifted")
    position = _masked_mean(F.smooth_l1_loss(predicted[:, :, :2], target[:, :, :2], reduction="none"), mask)
    velocity = _masked_mean(F.smooth_l1_loss(predicted[:, :, 2:], target[:, :, 2:], reduction="none"), mask)
    adjacency_float = adjacency.to(predicted.dtype)
    degree = adjacency_float.sum(dim=2, keepdim=True).clamp_min(1.0)
    predicted_neighbor = torch.bmm(adjacency_float, predicted[:, :, :2]) / degree
    target_neighbor = torch.bmm(adjacency_float, target[:, :, :2]) / degree
    graph = _masked_mean(
        F.smooth_l1_loss(
            predicted[:, :, :2] - predicted_neighbor,
            target[:, :, :2] - target_neighbor,
            reduction="none",
        ),
        mask,
    )
    acceleration = _masked_mean(
        F.smooth_l1_loss(
            predicted[:, :, 2:] - previous[:, :, 2:],
            target[:, :, 2:] - previous[:, :, 2:],
            reduction="none",
        ),
        mask,
    )
    outside = (predicted * (~mask)[:, :, None].to(predicted.dtype)).abs().max()
    total = position + velocity * 0.45 + graph * 0.30 + acceleration * 0.15 + outside
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("cellular motion loss became non-finite")
    return total, {
        "loss": total.detach(),
        "position": position.detach(),
        "velocity": velocity.detach(),
        "graph": graph.detach(),
        "acceleration": acceleration.detach(),
        "outside": outside.detach(),
    }
