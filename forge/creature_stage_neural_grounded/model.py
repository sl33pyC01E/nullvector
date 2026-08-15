from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ..creature_stage_neural_motion.contract import CellularMotionTransformerConfig
from ..creature_stage_neural_motion.model import CellularMotionTransformer
from .contract import DYNAMIC_FEATURES, MAX_CELLS, GroundedModelConfig


class ExpandedCellularMotionTransformer(CellularMotionTransformer):
    """The frozen transformer with only its sequence-length assertion expanded."""

    def forward(self, static: Tensor, state: Tensor, mask: Tensor, adjacency: Tensor,
                family: Tensor, morphotype: Tensor, motion: Tensor, phase: Tensor,
                controls: Tensor) -> Tensor:
        batch, cells = static.shape[:2]
        if (
            cells != MAX_CELLS
            or static.shape != (batch, cells, self.config.static_features)
            or state.shape != (batch, cells, self.config.state_features)
            or mask.shape != (batch, cells) or mask.dtype is not torch.bool
            or adjacency.shape != (batch, cells, cells) or adjacency.dtype is not torch.bool
            or not bool(mask.any(1).all())
            or bool((adjacency & ~(mask[:, :, None] & mask[:, None, :])).any())
            or not bool(torch.isfinite(static).all()) or not bool(torch.isfinite(state).all())
        ):
            raise ValueError("expanded grounded transformer input drifted")
        condition = self.condition(family, morphotype, motion, phase, controls)
        value = self.static_projection(static.float()) + self.state_projection(state.float()) + self.condition_projection(condition)[:, None]
        value = value * mask[:, :, None].to(value.dtype)
        for block in self.blocks:
            value = block(value, condition, mask, adjacency)
        return self.out(value) * mask[:, :, None].to(value.dtype)


class GroundedGraphBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.local = nn.Sequential(nn.Linear(width * 2, width * 2), nn.SiLU(), nn.Linear(width * 2, width))
        self.gate = nn.Sequential(nn.Linear(width * 2, width), nn.Sigmoid())

    def forward(self, value: Tensor, mask: Tensor, adjacency: Tensor) -> Tensor:
        normalized = self.norm(value)
        weights = adjacency.to(value.dtype)
        neighbor = torch.bmm(weights, normalized) / weights.sum(2, keepdim=True).clamp_min(1)
        joined = torch.cat((normalized, neighbor), dim=-1)
        value = value + self.local(joined) * self.gate(joined)
        return value * mask[:, :, None].to(value.dtype)


@dataclass(slots=True)
class GroundedMotionOutput:
    cells: Tensor
    body_velocity: Tensor
    direct_cells: Tensor
    delta_cells: Tensor
    blend: Tensor


class NeuralGroundedMotion(nn.Module):
    def __init__(self, backbone_config: CellularMotionTransformerConfig,
                 config: GroundedModelConfig = GroundedModelConfig()) -> None:
        super().__init__()
        self.backbone_config = backbone_config
        self.config = config
        self.backbone = ExpandedCellularMotionTransformer(backbone_config)
        input_width = 61 + 4 + DYNAMIC_FEATURES + 4
        width = config.refinement_width
        self.refine_in = nn.Sequential(nn.Linear(input_width, width), nn.SiLU(), nn.Linear(width, width))
        self.refine_blocks = nn.ModuleList(GroundedGraphBlock(width) for _ in range(config.refinement_depth))
        self.direct_head = nn.Sequential(nn.LayerNorm(width), nn.SiLU(), nn.Linear(width, 4), nn.Tanh())
        self.delta_head = nn.Sequential(nn.LayerNorm(width), nn.SiLU(), nn.Linear(width, 4), nn.Tanh())
        self.blend_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1), nn.Sigmoid())
        self.body_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1), nn.Tanh())

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, static: Tensor, state: Tensor, dynamic: Tensor, mask: Tensor,
                adjacency: Tensor, family: Tensor, morphotype: Tensor, motion: Tensor,
                phase: Tensor, controls: Tensor) -> GroundedMotionOutput:
        if dynamic.shape != (*mask.shape, DYNAMIC_FEATURES):
            raise ValueError("grounded dynamic feature shape drifted")
        prior = self.backbone(static, state, mask, adjacency, family, morphotype, motion, phase, controls)
        value = self.refine_in(torch.cat((static.float(), state.float(), dynamic.float(), prior.float()), dim=-1))
        value = value * mask[:, :, None].to(value.dtype)
        for block in self.refine_blocks:
            value = block(value, mask, adjacency)
        direct = self.direct_head(value)
        delta = self.delta_head(value) * .08
        blend = self.blend_head(value)
        position = blend * direct[:, :, :2] + (1.0 - blend) * (state[:, :, :2] + delta[:, :, :2])
        # Velocity is a derivative, not an accumulated state.  Treating it as
        # another recurrent displacement was the source of the old flailing.
        cells = torch.cat((position, direct[:, :, 2:]), dim=-1)
        cells = cells * mask[:, :, None].to(cells.dtype)
        active = mask[:, :, None].to(value.dtype)
        pooled = (value * active).sum(1) / active.sum(1).clamp_min(1)
        body = self.body_head(pooled)[:, 0]
        return GroundedMotionOutput(cells, body, direct, delta, blend)


def _masked_mean(value: Tensor, mask: Tensor) -> Tensor:
    active = mask[:, :, None].to(value.dtype)
    return (value * active).sum() / (active.sum().clamp_min(1) * value.shape[-1])


def grounded_loss(output: GroundedMotionOutput, batch: dict[str, Tensor], *,
                  position_weight: float = 1.0, velocity_weight: float = .35,
                  appendage_weight: float = .55, contact_weight: float = .65,
                  graph_weight: float = .25, body_velocity_weight: float = .55,
                  delta_weight: float = .20) -> tuple[Tensor, dict[str, Tensor]]:
    predicted, target, state, mask = output.cells.float(), batch["target"].float(), batch["state"].float(), batch["mask"]
    pos_error = F.smooth_l1_loss(predicted[:, :, :2], target[:, :, :2], reduction="none")
    vel_error = F.smooth_l1_loss(predicted[:, :, 2:], target[:, :, 2:], reduction="none")
    position = _masked_mean(pos_error, mask)
    velocity = _masked_mean(vel_error, mask)
    appendage = (batch["static"][:, :, 50] > .5) & mask
    appendage_loss = _masked_mean(pos_error, appendage)
    contact = (batch["dynamic"][:, :, 5] > .5) & mask
    contact_loss = _masked_mean(pos_error, contact) if bool(contact.any()) else position * 0
    adjacency = batch["adjacency"].to(predicted.dtype)
    degree = adjacency.sum(2, keepdim=True).clamp_min(1)
    pred_neighbor = torch.bmm(adjacency, predicted[:, :, :2]) / degree
    target_neighbor = torch.bmm(adjacency, target[:, :, :2]) / degree
    graph = _masked_mean(F.smooth_l1_loss(predicted[:, :, :2] - pred_neighbor, target[:, :, :2] - target_neighbor, reduction="none"), mask)
    delta = _masked_mean(F.smooth_l1_loss(output.delta_cells[:, :, :2], target[:, :, :2] - state[:, :, :2], reduction="none"), mask)
    direct = _masked_mean(F.smooth_l1_loss(output.direct_cells, target, reduction="none"), mask)
    blend_target = torch.full_like(output.blend, .62)
    blend_regularization = _masked_mean(F.smooth_l1_loss(output.blend, blend_target, reduction="none"), mask)
    body = F.smooth_l1_loss(output.body_velocity.float(), batch["body_target"].float())
    outside = output.cells[~mask].abs().max()
    total = position * position_weight + velocity * velocity_weight + appendage_loss * appendage_weight + contact_loss * contact_weight + graph * graph_weight + body * body_velocity_weight + delta * delta_weight + direct * .60 + blend_regularization * .12 + outside
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("grounded neural loss became non-finite")
    return total, {"loss": total.detach(), "position": position.detach(), "velocity": velocity.detach(), "appendage": appendage_loss.detach(), "contact": contact_loss.detach(), "graph": graph.detach(), "body_velocity": body.detach(), "delta": delta.detach(), "direct": direct.detach(), "blend_regularization": blend_regularization.detach(), "outside": outside.detach(), "blend": output.blend[mask].mean().detach()}
