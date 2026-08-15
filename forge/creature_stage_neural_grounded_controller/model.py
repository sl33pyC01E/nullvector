from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .contract import ControllerModelConfig, MAX_APPENDAGES, MAX_MUSCLES, MUSCLE_META_FEATURES, OWNER_META_FEATURES
from .dataset import POOLED_FEATURES


@dataclass(slots=True)
class ControllerOutput:
    muscle_activation: Tensor
    contact_logits: Tensor
    body_velocity: Tensor
    owner_latent: Tensor


class OwnerBlock(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.global_film = nn.Linear(width, width * 2)
        self.net = nn.Sequential(nn.Linear(width, width * 3), nn.SiLU(), nn.Dropout(dropout), nn.Linear(width * 3, width))

    def forward(self, value: Tensor, global_value: Tensor, mask: Tensor) -> Tensor:
        scale, shift = self.global_film(global_value).chunk(2, dim=-1)
        conditioned = self.norm(value) * (1 + scale[:, None]) + shift[:, None]
        return (value + self.net(conditioned)) * mask[:, :, None].to(value.dtype)


class NeuralGroundedController(nn.Module):
    """Neural contact and muscle policy executed by the grounded PBD solver."""

    def __init__(self, config: ControllerModelConfig = ControllerModelConfig()) -> None:
        super().__init__(); self.config = config; width = config.width
        self.owner_in = nn.Sequential(nn.Linear(POOLED_FEATURES + OWNER_META_FEATURES, width), nn.SiLU(), nn.Linear(width, width))
        self.global_in = nn.Sequential(nn.Linear(POOLED_FEATURES, width), nn.SiLU(), nn.Linear(width, width))
        self.blocks = nn.ModuleList(OwnerBlock(width, config.dropout) for _ in range(config.depth))
        self.contact_head = nn.Sequential(nn.LayerNorm(width), nn.SiLU(), nn.Linear(width, 1))
        self.muscle_head = nn.Sequential(nn.Linear(width + MUSCLE_META_FEATURES, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1))
        self.body_head = nn.Sequential(nn.LayerNorm(width), nn.SiLU(), nn.Linear(width, 1), nn.Tanh())

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, owner_input: Tensor, global_input: Tensor, owner_meta: Tensor, owner_mask: Tensor,
                muscle_meta: Tensor, muscle_owner: Tensor, muscle_mask: Tensor) -> ControllerOutput:
        batch = owner_input.shape[0]
        if (
            owner_input.shape != (batch, MAX_APPENDAGES, POOLED_FEATURES)
            or global_input.shape != (batch, POOLED_FEATURES)
            or owner_meta.shape != (batch, MAX_APPENDAGES, OWNER_META_FEATURES)
            or owner_mask.shape != (batch, MAX_APPENDAGES) or owner_mask.dtype is not torch.bool
            or muscle_meta.shape != (batch, MAX_MUSCLES, MUSCLE_META_FEATURES)
            or muscle_owner.shape != (batch, MAX_MUSCLES) or muscle_owner.dtype != torch.long
            or muscle_mask.shape != (batch, MAX_MUSCLES) or muscle_mask.dtype is not torch.bool
            or bool((muscle_owner < 0).any()) or bool((muscle_owner >= MAX_APPENDAGES).any())
            or not bool(torch.isfinite(owner_input).all()) or not bool(torch.isfinite(global_input).all())
        ):
            raise ValueError("neural grounded controller input drifted")
        global_value = self.global_in(global_input.float())
        value = self.owner_in(torch.cat((owner_input.float(), owner_meta.float()), dim=-1))
        value = value * owner_mask[:, :, None].to(value.dtype)
        for block in self.blocks:
            value = block(value, global_value, owner_mask)
        contact = self.contact_head(value)[:, :, 0]
        gathered = torch.gather(value, 1, muscle_owner[:, :, None].expand(-1, -1, value.shape[-1]))
        muscle = torch.sigmoid(self.muscle_head(torch.cat((gathered, muscle_meta.float()), dim=-1))[:, :, 0])
        muscle = muscle * muscle_mask.to(muscle.dtype)
        contact = contact.masked_fill(~owner_mask, -30.0)
        return ControllerOutput(muscle, contact, self.body_head(global_value)[:, 0], value)
