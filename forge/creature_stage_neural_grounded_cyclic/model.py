from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

from ..creature_stage_neural_grounded.model import ExpandedCellularMotionTransformer, GroundedGraphBlock
from ..creature_stage_neural_motion.contract import CellularMotionTransformerConfig
from .contract import DYNAMIC_FEATURES, STATIC_FEATURES, STATE_FEATURES, CyclicModelConfig


def cyclic_phase_features(phase: Tensor, harmonics: int) -> Tensor:
    if phase.ndim != 1 or not bool(torch.isfinite(phase).all()):
        raise ValueError("cyclic phase input drifted")
    frequency = torch.arange(1, harmonics + 1, dtype=torch.float32, device=phase.device)[None]
    # Canonicalize phase before trigonometry so phase 0 and phase 1 are not
    # merely close after floating-point sin/cos; they are byte-identical.
    angle = torch.remainder(phase.float(), 1.0)[:, None] * math.tau * frequency
    return torch.cat((angle.sin(), angle.cos()), dim=1)


@dataclass(slots=True)
class CyclicMotionOutput:
    cells: Tensor
    body_velocity: Tensor
    direct_cells: Tensor
    recurrent_cells: Tensor
    direct_gate: Tensor


class NeuralCyclicGroundedMotion(nn.Module):
    """Periodic direct pose field with a tightly bounded recurrent correction."""

    def __init__(self, backbone_config: CellularMotionTransformerConfig,
                 config: CyclicModelConfig = CyclicModelConfig()) -> None:
        super().__init__()
        self.backbone_config = backbone_config
        self.config = config
        self.backbone = ExpandedCellularMotionTransformer(backbone_config)
        input_width = STATIC_FEATURES + STATE_FEATURES + DYNAMIC_FEATURES + 4 + config.harmonics * 2
        width = config.refinement_width
        self.refine_in = nn.Sequential(nn.Linear(input_width, width), nn.SiLU(), nn.Linear(width, width))
        self.refine_blocks = nn.ModuleList(GroundedGraphBlock(width) for _ in range(config.refinement_depth))
        self.direct_head = nn.Sequential(nn.LayerNorm(width), nn.SiLU(), nn.Linear(width, 4), nn.Tanh())
        self.recurrent_head = nn.Sequential(nn.LayerNorm(width), nn.SiLU(), nn.Linear(width, 2), nn.Tanh())
        self.gate_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1), nn.Sigmoid())
        self.body_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1), nn.Tanh())

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, static: Tensor, state: Tensor, dynamic: Tensor, mask: Tensor,
                adjacency: Tensor, family: Tensor, morphotype: Tensor, motion: Tensor,
                phase: Tensor, controls: Tensor) -> CyclicMotionOutput:
        if dynamic.shape != (*mask.shape, DYNAMIC_FEATURES):
            raise ValueError("cyclic grounded dynamic feature shape drifted")
        prior = self.backbone(static, state, mask, adjacency, family, morphotype, motion, phase, controls)
        phase_features = cyclic_phase_features(phase, self.config.harmonics)
        phase_cells = phase_features[:, None].expand(-1, mask.shape[1], -1)
        value = self.refine_in(torch.cat((static.float(), state.float(), dynamic.float(), prior.float(), phase_cells), dim=-1))
        value = value * mask[:, :, None].to(value.dtype)
        for block in self.refine_blocks:
            value = block(value, mask, adjacency)
        direct = self.direct_head(value)
        recurrent = state[:, :, :2].float() + self.recurrent_head(value) * self.config.recurrent_scale
        gate = self.config.direct_floor + (1.0 - self.config.direct_floor) * self.gate_head(value)
        position = gate * direct[:, :, :2] + (1.0 - gate) * recurrent
        cells = torch.cat((position, direct[:, :, 2:]), dim=-1)
        active = mask[:, :, None].to(cells.dtype)
        cells = cells * active
        pooled = (value * active).sum(1) / active.sum(1).clamp_min(1)
        return CyclicMotionOutput(cells, self.body_head(pooled)[:, 0], direct, recurrent, gate)
