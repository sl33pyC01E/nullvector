from __future__ import annotations

from typing import TypedDict

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ..creature_stage_developmental_motion.contract import (
    MAX_CELLS,
    MAX_DISPLACEMENT,
    MAX_MUSCLES,
    MAX_NODES,
    MUSCLE_FEATURES,
    NODE_FEATURES,
)
from ..creature_stage_developmental_motion.model import (
    CellGraphBlock,
    ConditionedTransformerBlock,
    DevelopmentalConditionEncoder,
    developmental_actuator_loss,
    masked_mean,
)
from .contract import CausalActuatorConfig, CausalTrainingConfig


class CausalActuatorOutput(TypedDict):
    cell_state: Tensor
    node_state: Tensor
    muscle_activation: Tensor
    muscle_node_force: Tensor
    parent_prior: Tensor
    skinned_state: Tensor


def muscle_node_force(
    muscle_features: Tensor,
    activation: Tensor,
    muscle_mask: Tensor,
    incidence: Tensor,
) -> Tensor:
    """Differentiable antagonistic force in normalized rest-coordinate space."""

    rest_delta = muscle_features[:, :, 2:4].float() - muscle_features[:, :, 0:2].float()
    normal = torch.stack((-rest_delta[:, :, 1], rest_delta[:, :, 0]), dim=2)
    normal = normal / normal.norm(dim=2, keepdim=True).clamp_min(1e-6)
    strength = muscle_features[:, :, 5:6].float().abs().clamp(0.0, 1.0)
    force = normal * activation.float()[:, :, None] * strength
    force = force * muscle_mask[:, :, None].to(force.dtype)
    # Signed incidence already encodes origin/insertion and antagonist sign.
    return torch.bmm(incidence.float().transpose(1, 2), force)


class MuscleCausalSkeletonActuator(nn.Module):
    """Predict contraction first, then make that contraction move the joints."""

    def __init__(self, config: CausalActuatorConfig | None = None) -> None:
        super().__init__()
        self.config = config or CausalActuatorConfig()
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
        self.force_to_node = nn.Sequential(
            nn.Linear(3, self.config.width), nn.SiLU(), nn.Linear(self.config.width, self.config.width)
        )
        self.previous_muscle_gate = nn.Parameter(torch.tensor(self.config.initial_previous_muscle_gate))
        self.force_gate = nn.Parameter(torch.tensor(self.config.initial_force_gate))

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
    ) -> tuple[Tensor, Tensor, Tensor]:
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
            raise ValueError("muscle-causal skeleton tensor interface drifted")
        condition = self.condition(family, morphotype, phase, traits)
        recurrence = torch.sigmoid(self.previous_muscle_gate)
        muscle_tokens = (
            self.muscle_features(muscle_features.float())
            + self.muscle_state(muscle_state.float()[:, :, None]) * recurrence
            + self.condition_to_width(condition)[:, None]
        ) * muscle_mask[:, :, None].to(node_features.dtype)
        absolute_incidence = muscle_incidence.abs().to(muscle_tokens.dtype)
        muscle_node = torch.bmm(absolute_incidence.transpose(1, 2), muscle_tokens)
        muscle_node /= absolute_incidence.transpose(1, 2).sum(dim=2, keepdim=True).clamp_min(1.0)
        value = (
            self.node_features(node_features.float())
            + self.node_state(node_state.float())
            + self.parent_node(parent_node.float())
            + self.muscle_to_node(muscle_node)
            + self.condition_to_width(condition)[:, None]
        ) * node_mask[:, :, None].to(node_features.dtype)
        for block in self.blocks:
            value = block(value, condition, node_mask, node_adjacency)

        node_to_muscle = torch.bmm(absolute_incidence, value)
        node_to_muscle /= absolute_incidence.sum(dim=2, keepdim=True).clamp_min(1.0)
        predicted_muscle = torch.sigmoid(self.muscle_out(muscle_tokens + node_to_muscle).squeeze(-1))
        predicted_muscle = predicted_muscle * muscle_mask.to(predicted_muscle.dtype)

        force = muscle_node_force(muscle_features, predicted_muscle, muscle_mask, muscle_incidence)
        force_features = torch.cat((force, force.norm(dim=2, keepdim=True)), dim=2)
        driven = value + self.force_to_node(force_features) * torch.sigmoid(self.force_gate)
        predicted_node = self.node_out(driven)
        direct = torch.cat((force, torch.zeros_like(force)), dim=2) * self.config.direct_force_scale
        predicted_node = torch.tanh(predicted_node + direct)
        predicted_node = predicted_node * node_mask[:, :, None].to(predicted_node.dtype)
        force = force * node_mask[:, :, None].to(force.dtype)
        return predicted_node, predicted_muscle, force


class MuscleCausalCellularActuator(nn.Module):
    """Same reviewed body interface as v1, with a causal muscle-to-joint path."""

    def __init__(self, config: CausalActuatorConfig | None = None) -> None:
        super().__init__()
        self.config = config or CausalActuatorConfig()
        self.actuator = MuscleCausalSkeletonActuator(self.config)
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
    ) -> CausalActuatorOutput:
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
            raise ValueError("muscle-causal cellular input contract drifted")
        weights = cell_node_weights.to(parent_prior.dtype)
        node_denominator = weights.sum(dim=1).clamp_min(1e-6)[:, :, None]
        parent_node = torch.bmm(weights.transpose(1, 2), parent_prior) / node_denominator
        parent_full = parent_prior * mask[:, :, None].to(parent_prior.dtype)
        predicted_node, predicted_muscle, muscle_force = self.actuator(
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
            "muscle_node_force": muscle_force,
            "parent_prior": parent_full,
            "skinned_state": skinned,
        }


def causal_actuator_loss(
    output: CausalActuatorOutput,
    frame: dict[str, Tensor],
    previous_cell: Tensor,
    previous_node: Tensor,
    previous_muscle: Tensor,
    config: CausalTrainingConfig,
) -> tuple[Tensor, dict[str, Tensor]]:
    base, pieces = developmental_actuator_loss(output, frame, previous_cell, previous_node, config)
    muscle = output["muscle_activation"].float()
    target = frame["muscle_target"].float()
    muscle_mask = frame["muscle_mask"]
    muscle_l1 = masked_mean((muscle - target).abs(), muscle_mask)
    target_velocity = target - frame["muscle_state"].float()
    predicted_velocity = muscle - previous_muscle.float()
    muscle_velocity = masked_mean(
        F.smooth_l1_loss(predicted_velocity, target_velocity, reduction="none"), muscle_mask
    )
    target_force = muscle_node_force(
        frame["muscle_features"], target, muscle_mask, frame["muscle_incidence"]
    )
    force_loss = masked_mean(
        F.smooth_l1_loss(output["muscle_node_force"].float(), target_force, reduction="none"),
        frame["node_mask"],
    )
    total = (
        base + muscle_l1 * config.muscle_l1_weight
        + muscle_velocity * config.muscle_velocity_weight
        + force_loss * config.muscle_force_weight
    )
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("muscle-causal actuator loss became non-finite")
    pieces = dict(pieces)
    pieces.update({
        "loss": total.detach(), "muscle_l1": muscle_l1.detach(),
        "muscle_velocity": muscle_velocity.detach(), "muscle_force": force_loss.detach(),
    })
    return total, pieces
