from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ..creature_stage_neural_grounded_cyclic.model import CyclicMotionOutput, NeuralCyclicGroundedMotion
from ..creature_stage_neural_grounded_cyclic.contract import CyclicModelConfig
from ..creature_stage_neural_motion.contract import CellularMotionTransformerConfig
from .contract import MAX_APPENDAGES, ComponentModelConfig


@dataclass(slots=True)
class ComponentMotionOutput:
    cells: Tensor
    body_velocity: Tensor
    base: CyclicMotionOutput
    owner_translation: Tensor
    local_correction: Tensor


class ComponentBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.net = nn.Sequential(nn.Linear(width * 2, width * 2), nn.SiLU(), nn.Linear(width * 2, width))
        self.gate = nn.Sequential(nn.Linear(width * 2, width), nn.Sigmoid())

    def forward(self, value: Tensor, owner_context: Tensor, mask: Tensor) -> Tensor:
        joined = torch.cat((self.norm(value), owner_context), dim=-1)
        return (value + self.net(joined) * self.gate(joined)) * mask[:, :, None].to(value.dtype)


class NeuralComponentGroundedMotion(nn.Module):
    """Adds appendage-wide neural transforms to the cyclic cell controller."""

    def __init__(self, backbone_config: CellularMotionTransformerConfig,
                 cyclic_config: CyclicModelConfig, config: ComponentModelConfig = ComponentModelConfig()) -> None:
        super().__init__()
        self.config = config
        self.base = NeuralCyclicGroundedMotion(backbone_config, cyclic_config)
        input_width = 61 + 4 + 16 + 4
        self.cell_in = nn.Sequential(nn.Linear(input_width, config.width), nn.SiLU(), nn.Linear(config.width, config.width))
        self.owner_in = nn.Sequential(nn.Linear(config.width + 16, config.width), nn.SiLU(), nn.Linear(config.width, config.width))
        self.blocks = nn.ModuleList(ComponentBlock(config.width) for _ in range(config.depth))
        self.translation_head = nn.Sequential(nn.LayerNorm(config.width), nn.SiLU(), nn.Linear(config.width, 2), nn.Tanh())
        self.local_head = nn.Sequential(nn.LayerNorm(config.width), nn.SiLU(), nn.Linear(config.width, 4), nn.Tanh())
        self.body_head = nn.Sequential(nn.LayerNorm(config.width), nn.Linear(config.width, 1), nn.Tanh())

    @property
    def parameter_count(self) -> int: return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, static: Tensor, state: Tensor, dynamic: Tensor, owner: Tensor, mask: Tensor,
                adjacency: Tensor, family: Tensor, morphotype: Tensor, motion: Tensor,
                phase: Tensor, controls: Tensor) -> ComponentMotionOutput:
        if owner.shape != mask.shape or owner.dtype != torch.long or bool((owner < -1).any()) or bool((owner >= MAX_APPENDAGES).any()):
            raise ValueError("component motion owner input drifted")
        base = self.base(static, state, dynamic, mask, adjacency, family, morphotype, motion, phase, controls)
        value = self.cell_in(torch.cat((static.float(), state.float(), dynamic.float(), base.cells.float()), dim=-1))
        value = value * mask[:, :, None].to(value.dtype)
        owner_one_hot = torch.nn.functional.one_hot(owner.clamp_min(0), MAX_APPENDAGES).to(value.dtype)
        owner_one_hot = owner_one_hot * (owner >= 0)[:, :, None]
        counts = owner_one_hot.sum(1).clamp_min(1)
        pooled = torch.bmm(owner_one_hot.transpose(1, 2), value) / counts[:, :, None]
        # Pool the known anchor/contact channels alongside anatomy so a shared
        # appendage transform can be inferred before returning to cell detail.
        pooled_dynamic = torch.bmm(owner_one_hot.transpose(1, 2), dynamic.float()) / counts[:, :, None]
        owner_latent = self.owner_in(torch.cat((pooled, pooled_dynamic), dim=-1))
        owner_context = torch.bmm(owner_one_hot, owner_latent)
        for block in self.blocks: value = block(value, owner_context, mask)
        owner_translation = self.translation_head(owner_latent) * self.config.translation_scale
        broadcast_translation = torch.bmm(owner_one_hot, owner_translation)
        local = self.local_head(value) * self.config.local_scale
        cells = base.cells.float() + local
        cells[:, :, :2] += broadcast_translation
        cells = cells * mask[:, :, None].to(cells.dtype)
        active = mask[:, :, None].to(value.dtype); pooled_body = (value * active).sum(1) / active.sum(1).clamp_min(1)
        body = (base.body_velocity.float() + self.body_head(pooled_body)[:, 0] * .08).clamp(-1, 1)
        return ComponentMotionOutput(cells, body, base, owner_translation, local)
