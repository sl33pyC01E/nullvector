from __future__ import annotations

import math
from typing import TypedDict

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ..creature_stage_developmental.contract import TRAITS
from .contract import (
    MAX_CELLS,
    MAX_DISPLACEMENT,
    MAX_MUSCLES,
    MAX_NODES,
    MUSCLE_FEATURES,
    NODE_FEATURES,
    DevelopmentalActuatorConfig,
)


def _phase_features(phase: Tensor, width: int) -> Tensor:
    if phase.ndim != 1 or width < 16 or not bool(torch.isfinite(phase).all()):
        raise ValueError("developmental actuator phase input drifted")
    quarter = width // 4
    frequencies = torch.arange(1, quarter + 1, dtype=torch.float32, device=phase.device)[None]
    angle = phase.float()[:, None] * math.tau * frequencies
    encoded = torch.cat((angle.sin(), angle.cos(), (angle * .5).sin(), (angle * .5).cos()), dim=1)
    return F.pad(encoded, (0, width - encoded.shape[1]))


class ActuatorOutput(TypedDict):
    cell_state: Tensor
    node_state: Tensor
    muscle_activation: Tensor
    parent_prior: Tensor
    skinned_state: Tensor


class DevelopmentalConditionEncoder(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.family = nn.Embedding(5, width)
        self.morphotype = nn.Embedding(20, width)
        self.phase = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.traits = nn.Sequential(nn.Linear(len(TRAITS), width), nn.SiLU(), nn.Linear(width, width))
        self.out = nn.Sequential(nn.LayerNorm(width), nn.SiLU(), nn.Linear(width, width))

    def forward(self, family: Tensor, morphotype: Tensor, phase: Tensor, traits: Tensor) -> Tensor:
        batch = family.shape[0]
        if (
            family.shape != (batch,) or morphotype.shape != (batch,) or phase.shape != (batch,)
            or traits.shape != (batch, len(TRAITS))
            or bool((family < 0).any()) or bool((family >= 5).any())
            or bool((morphotype < 0).any()) or bool((morphotype >= 20).any())
            or not bool(torch.isfinite(traits).all())
        ):
            raise ValueError("developmental actuator condition contract drifted")
        value = (
            self.family(family)
            + self.morphotype(morphotype)
            + self.phase(_phase_features(phase, self.width))
            + self.traits(traits.float())
        )
        return self.out(value)


class ConditionedTransformerBlock(nn.Module):
    def __init__(self, config: DevelopmentalActuatorConfig) -> None:
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
    def _film(norm: nn.LayerNorm, layer: nn.Linear, value: Tensor, condition: Tensor) -> Tensor:
        scale, shift = layer(condition).chunk(2, dim=1)
        return norm(value) * (1.0 + scale[:, None]) + shift[:, None]

    def forward(self, value: Tensor, condition: Tensor, mask: Tensor, adjacency: Tensor) -> Tensor:
        attended_input = self._film(self.norm_attention, self.attention_film, value, condition)
        attended, _ = self.attention(
            attended_input, attended_input, attended_input,
            key_padding_mask=~mask, need_weights=False,
        )
        value = value + self.dropout(attended)
        graph = adjacency.to(value.dtype)
        neighbors = torch.bmm(graph, self.norm_graph(value)) / graph.sum(dim=2, keepdim=True).clamp_min(1.0)
        value = value + self.dropout(self.graph(neighbors))
        fed = self._film(self.norm_ff, self.ff_film, value, condition)
        value = value + self.dropout(self.feedforward(fed))
        return value * mask[:, :, None].to(value.dtype)


class CellGraphBlock(nn.Module):
    def __init__(self, width: int, condition_width: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.film = nn.Linear(condition_width, width * 2)
        self.self_layer = nn.Linear(width, width)
        self.neighbor_layer = nn.Linear(width, width)
        self.out = nn.Sequential(nn.SiLU(), nn.Dropout(dropout), nn.Linear(width, width))

    def forward(self, value: Tensor, condition: Tensor, mask: Tensor, adjacency: Tensor) -> Tensor:
        scale, shift = self.film(condition).chunk(2, dim=1)
        normalized = self.norm(value) * (1.0 + scale[:, None]) + shift[:, None]
        graph = adjacency.to(value.dtype)
        neighbor = torch.bmm(graph, normalized) / graph.sum(dim=2, keepdim=True).clamp_min(1.0)
        value = value + self.out(self.self_layer(normalized) + self.neighbor_layer(neighbor))
        return value * mask[:, :, None].to(value.dtype)


class DevelopmentalActuatorMotionTransformer(nn.Module):
    """Skeleton/muscle controller; the cell head cannot move anatomy independently."""

    def __init__(self, config: DevelopmentalActuatorConfig | None = None) -> None:
        super().__init__()
        self.config = config or DevelopmentalActuatorConfig()
        self.condition = DevelopmentalConditionEncoder(self.config.condition_width)
        self.condition_to_width = nn.Linear(self.config.condition_width, self.config.width)
        self.node_features = nn.Linear(NODE_FEATURES, self.config.width)
        self.node_state = nn.Linear(4, self.config.width)
        self.parent_node = nn.Linear(4, self.config.width)
        self.muscle_features = nn.Linear(MUSCLE_FEATURES, self.config.width)
        self.muscle_state = nn.Linear(1, self.config.width)
        self.muscle_to_node = nn.Linear(self.config.width, self.config.width)
        self.blocks = nn.ModuleList(ConditionedTransformerBlock(self.config) for _ in range(self.config.depth))
        self.node_out = nn.Sequential(
            nn.LayerNorm(self.config.width), nn.SiLU(), nn.Linear(self.config.width, 4), nn.Tanh()
        )
        self.muscle_out = nn.Sequential(
            nn.LayerNorm(self.config.width), nn.SiLU(), nn.Linear(self.config.width, 1)
        )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(
        self,
        node_features: Tensor,
        node_state: Tensor,
        node_mask: Tensor,
        node_adjacency: Tensor,
        muscle_features: Tensor,
        muscle_state: Tensor,
        muscle_mask: Tensor,
        muscle_incidence: Tensor,
        parent_node: Tensor,
        family: Tensor,
        morphotype: Tensor,
        phase: Tensor,
        traits: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch = node_features.shape[0]
        if (
            node_features.shape != (batch, MAX_NODES, NODE_FEATURES)
            or node_state.shape != (batch, MAX_NODES, 4)
            or node_mask.shape != (batch, MAX_NODES) or node_mask.dtype is not torch.bool
            or node_adjacency.shape != (batch, MAX_NODES, MAX_NODES) or node_adjacency.dtype is not torch.bool
            or muscle_features.shape != (batch, MAX_MUSCLES, MUSCLE_FEATURES)
            or muscle_state.shape != (batch, MAX_MUSCLES)
            or muscle_mask.shape != (batch, MAX_MUSCLES) or muscle_mask.dtype is not torch.bool
            or muscle_incidence.shape != (batch, MAX_MUSCLES, MAX_NODES)
            or parent_node.shape != (batch, MAX_NODES, 4)
        ):
            raise ValueError("developmental actuator tensor interface drifted")
        condition = self.condition(family, morphotype, phase, traits)
        muscle_tokens = (
            self.muscle_features(muscle_features.float())
            + self.muscle_state(muscle_state.float()[:, :, None])
            + self.condition_to_width(condition)[:, None]
        ) * muscle_mask[:, :, None].to(node_features.dtype)
        incidence = muscle_incidence.abs().to(muscle_tokens.dtype)
        muscle_node = torch.bmm(incidence.transpose(1, 2), muscle_tokens)
        muscle_node /= incidence.transpose(1, 2).sum(dim=2, keepdim=True).clamp_min(1.0)
        value = (
            self.node_features(node_features.float())
            + self.node_state(node_state.float())
            + self.parent_node(parent_node.float())
            + self.muscle_to_node(muscle_node)
            + self.condition_to_width(condition)[:, None]
        ) * node_mask[:, :, None].to(node_features.dtype)
        for block in self.blocks:
            value = block(value, condition, node_mask, node_adjacency)
        predicted_node = self.node_out(value) * node_mask[:, :, None].to(value.dtype)
        node_to_muscle = torch.bmm(incidence, value)
        node_to_muscle /= incidence.sum(dim=2, keepdim=True).clamp_min(1.0)
        predicted_muscle = torch.sigmoid(self.muscle_out(muscle_tokens + node_to_muscle).squeeze(-1))
        predicted_muscle = predicted_muscle * muscle_mask.to(predicted_muscle.dtype)
        return predicted_node, predicted_muscle


class DevelopmentalCellularMotionTransformer(nn.Module):
    """Rollout-1000 prior distillation + neural actuator + local flesh residual."""

    def __init__(self, config: DevelopmentalActuatorConfig | None = None) -> None:
        super().__init__()
        self.config = config or DevelopmentalActuatorConfig()
        self.actuator = DevelopmentalActuatorMotionTransformer(self.config)
        self.cell_static = nn.Linear(61, self.config.cell_width)
        self.cell_state = nn.Linear(4, self.config.cell_width)
        self.cell_skinned = nn.Linear(4, self.config.cell_width)
        self.cell_parent = nn.Linear(4, self.config.cell_width)
        self.cell_condition = nn.Linear(self.config.condition_width, self.config.cell_width)
        self.cell_blocks = nn.ModuleList(
            CellGraphBlock(self.config.cell_width, self.config.condition_width, self.config.dropout)
            for _ in range(self.config.cell_graph_blocks)
        )
        self.cell_out = nn.Sequential(
            nn.LayerNorm(self.config.cell_width), nn.SiLU(), nn.Linear(self.config.cell_width, 4), nn.Tanh()
        )
        self.residual_gate = nn.Parameter(torch.tensor(-1.75))
        self.parent_gate = nn.Parameter(torch.tensor(-2.50))

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def trainable_parameter_count(self) -> int:
        return self.parameter_count

    def forward(
        self,
        static: Tensor,
        state: Tensor,
        mask: Tensor,
        adjacency: Tensor,
        node_features: Tensor,
        node_state: Tensor,
        node_mask: Tensor,
        node_adjacency: Tensor,
        muscle_features: Tensor,
        muscle_state: Tensor,
        muscle_mask: Tensor,
        muscle_incidence: Tensor,
        cell_node_weights: Tensor,
        parent_prior: Tensor,
        family: Tensor,
        morphotype: Tensor,
        phase: Tensor,
        traits: Tensor,
    ) -> ActuatorOutput:
        batch = static.shape[0]
        if (
            static.shape != (batch, MAX_CELLS, 61) or state.shape != (batch, MAX_CELLS, 4)
            or mask.shape != (batch, MAX_CELLS) or mask.dtype is not torch.bool
            or adjacency.shape != (batch, MAX_CELLS, MAX_CELLS) or adjacency.dtype is not torch.bool
            or cell_node_weights.shape != (batch, MAX_CELLS, MAX_NODES)
            or parent_prior.shape != (batch, MAX_CELLS, 4)
            or not bool(torch.isfinite(static).all()) or not bool(torch.isfinite(state).all())
            or not bool(torch.isfinite(parent_prior).all())
        ):
            raise ValueError("developmental cellular actuator input contract drifted")
        weights = cell_node_weights.to(parent_prior.dtype)
        node_denominator = weights.sum(dim=1).clamp_min(1e-6)[:, :, None]
        parent_node = torch.bmm(weights.transpose(1, 2), parent_prior) / node_denominator
        parent_full = parent_prior * mask[:, :, None].to(parent_prior.dtype)
        predicted_node, predicted_muscle = self.actuator(
            node_features, node_state, node_mask, node_adjacency,
            muscle_features, muscle_state, muscle_mask, muscle_incidence,
            parent_node, family, morphotype, phase, traits,
        )
        skinned = torch.bmm(weights, predicted_node)
        condition = self.actuator.condition(family, morphotype, phase, traits)
        cell = (
            self.cell_static(static.float()) + self.cell_state(state.float())
            + self.cell_skinned(skinned.float()) + self.cell_parent(parent_full.float())
            + self.cell_condition(condition)[:, None]
        ) * mask[:, :, None].to(static.dtype)
        for block in self.cell_blocks:
            cell = block(cell, condition, mask, adjacency)
        residual = self.cell_out(cell)
        residual_scale = torch.sigmoid(self.residual_gate) * .35
        parent_scale = torch.sigmoid(self.parent_gate) * .20
        predicted_cell = torch.tanh(skinned + residual * residual_scale + parent_full * parent_scale)
        predicted_cell = predicted_cell * mask[:, :, None].to(predicted_cell.dtype)
        return {
            "cell_state": predicted_cell,
            "node_state": predicted_node,
            "muscle_activation": predicted_muscle,
            "parent_prior": parent_full,
            "skinned_state": skinned,
        }


def masked_mean(value: Tensor, mask: Tensor) -> Tensor:
    active = mask.to(value.dtype)
    while active.ndim < value.ndim:
        active = active.unsqueeze(-1)
    expanded = active.expand_as(value)
    return (value * expanded).sum() / expanded.sum().clamp_min(1.0)


def developmental_actuator_loss(
    output: ActuatorOutput,
    frame: dict[str, Tensor],
    previous_cell: Tensor,
    previous_node: Tensor,
    config,
) -> tuple[Tensor, dict[str, Tensor]]:
    cell = output["cell_state"].float()
    node = output["node_state"].float()
    muscle = output["muscle_activation"].float()
    target = frame["target"].float()
    node_target = frame["node_target"].float()
    muscle_target = frame["muscle_target"].float()
    cell_mask = frame["mask"]
    node_mask = frame["node_mask"]
    muscle_mask = frame["muscle_mask"]
    cell_position = masked_mean(F.smooth_l1_loss(cell[:, :, :2], target[:, :, :2], reduction="none"), cell_mask)
    cell_velocity = masked_mean(F.smooth_l1_loss(cell[:, :, 2:], target[:, :, 2:], reduction="none"), cell_mask)
    node_position = masked_mean(F.smooth_l1_loss(node[:, :, :2], node_target[:, :, :2], reduction="none"), node_mask)
    node_velocity = masked_mean(F.smooth_l1_loss(node[:, :, 2:], node_target[:, :, 2:], reduction="none"), node_mask)
    muscle_loss = masked_mean(F.smooth_l1_loss(muscle, muscle_target, reduction="none"), muscle_mask)

    rest = frame["node_rest"].float()
    predicted_absolute = rest + node[:, :, :2] * MAX_DISPLACEMENT
    target_absolute = rest + node_target[:, :, :2] * MAX_DISPLACEMENT
    predicted_distance = torch.cdist(predicted_absolute, predicted_absolute)
    target_distance = torch.cdist(target_absolute, target_absolute)
    edge_mask = frame["node_adjacency"] & node_mask[:, :, None] & node_mask[:, None, :]
    identity = torch.eye(MAX_NODES, dtype=torch.bool, device=edge_mask.device)[None]
    edge_mask &= ~identity
    bone_length = masked_mean(
        F.smooth_l1_loss(predicted_distance / MAX_DISPLACEMENT, target_distance / MAX_DISPLACEMENT, reduction="none"),
        edge_mask,
    )
    appendage_mask = (frame["static"][:, :, 50] > .5) & cell_mask
    appendage = masked_mean(
        F.smooth_l1_loss(cell[:, :, :2], target[:, :, :2], reduction="none"), appendage_mask
    )
    target_change = (target[:, :, :2] - previous_cell[:, :, :2]).norm(dim=2)
    predicted_change = (cell[:, :, :2] - previous_cell[:, :, :2]).norm(dim=2)
    anti_copy = masked_mean(F.relu(target_change * .70 - predicted_change), cell_mask)
    acceleration = masked_mean(
        F.smooth_l1_loss(cell[:, :, 2:] - previous_cell[:, :, 2:], target[:, :, 2:] - frame["state"][:, :, 2:].float(), reduction="none"),
        cell_mask,
    )
    parent_prior = masked_mean(
        F.smooth_l1_loss(cell[:, :, :2], output["parent_prior"][:, :, :2].float(), reduction="none"),
        cell_mask,
    )
    outside = (cell * (~cell_mask)[:, :, None].to(cell.dtype)).abs().max()
    outside = outside + (node * (~node_mask)[:, :, None].to(node.dtype)).abs().max()
    outside = outside + (muscle * (~muscle_mask).to(muscle.dtype)).abs().max()
    total = (
        cell_position * config.cell_position_weight
        + cell_velocity * config.cell_velocity_weight
        + node_position * config.node_position_weight
        + node_velocity * config.node_velocity_weight
        + muscle_loss * config.muscle_weight
        + bone_length * config.bone_length_weight
        + appendage * config.appendage_weight
        + anti_copy * config.anti_copy_weight
        + acceleration * config.acceleration_weight
        + parent_prior * config.parent_prior_weight
        + outside
    )
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("developmental actuator loss became non-finite")
    return total, {
        "loss": total.detach(), "cell_position": cell_position.detach(),
        "cell_velocity": cell_velocity.detach(), "node_position": node_position.detach(),
        "node_velocity": node_velocity.detach(), "muscle": muscle_loss.detach(),
        "bone_length": bone_length.detach(), "appendage": appendage.detach(),
        "anti_copy": anti_copy.detach(), "acceleration": acceleration.detach(),
        "parent_prior": parent_prior.detach(), "outside": outside.detach(),
    }
