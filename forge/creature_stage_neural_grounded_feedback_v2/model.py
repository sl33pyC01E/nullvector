from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .contract import GLOBAL_FEATURES, MAX_APPENDAGES, MAX_MUSCLES, MUSCLE_FEATURES, OWNER_FEATURES, ModelConfig


@dataclass(slots=True)
class FeedbackOutput:
    muscle_activation: Tensor
    contact_logits: Tensor
    body_velocity: Tensor


class FeedbackBlock(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.film = nn.Linear(width, width * 2)
        self.net = nn.Sequential(nn.Linear(width, width * 3), nn.SiLU(), nn.Dropout(dropout), nn.Linear(width * 3, width))

    def forward(self, owners: Tensor, context: Tensor, mask: Tensor) -> Tensor:
        scale, shift = self.film(context).chunk(2, dim=-1)
        value = self.norm(owners) * (1 + scale[:, None]) + shift[:, None]
        pooled = (owners * mask[:, :, None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        return (owners + self.net(value + pooled[:, None])) * mask[:, :, None]


class NeuralGroundedFeedback(nn.Module):
    """Live-state neural muscle and contact policy executed through PBD."""

    def __init__(self, config: ModelConfig = ModelConfig()) -> None:
        super().__init__(); self.config = config; width = config.width
        self.owner_in = nn.Sequential(nn.Linear(OWNER_FEATURES, width), nn.SiLU(), nn.Linear(width, width))
        self.global_in = nn.Sequential(nn.Linear(GLOBAL_FEATURES, width), nn.SiLU(), nn.Linear(width, width))
        self.blocks = nn.ModuleList(FeedbackBlock(width, config.dropout) for _ in range(config.depth))
        self.contact_head = nn.Sequential(nn.LayerNorm(width), nn.SiLU(), nn.Linear(width, 1))
        self.muscle_head = nn.Sequential(nn.Linear(width + MUSCLE_FEATURES, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1))
        self.body_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1), nn.Tanh())

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, owner_state: Tensor, global_state: Tensor, owner_mask: Tensor,
                muscle_meta: Tensor, muscle_owner: Tensor, muscle_mask: Tensor) -> FeedbackOutput:
        batch = owner_state.shape[0]
        if (
            owner_state.shape != (batch, MAX_APPENDAGES, OWNER_FEATURES)
            or global_state.shape != (batch, GLOBAL_FEATURES)
            or owner_mask.shape != (batch, MAX_APPENDAGES) or owner_mask.dtype is not torch.bool
            or muscle_meta.shape != (batch, MAX_MUSCLES, MUSCLE_FEATURES)
            or muscle_owner.shape != (batch, MAX_MUSCLES) or muscle_owner.dtype != torch.long
            or muscle_mask.shape != (batch, MAX_MUSCLES) or muscle_mask.dtype is not torch.bool
            or bool((muscle_owner < 0).any()) or bool((muscle_owner >= MAX_APPENDAGES).any())
            or not bool(torch.isfinite(owner_state).all()) or not bool(torch.isfinite(global_state).all())
        ):
            raise ValueError("grounded feedback input drifted")
        mask = owner_mask[:, :, None].to(torch.float32)
        context = self.global_in(global_state.float())
        owners = self.owner_in(owner_state.float()) * mask
        for block in self.blocks:
            owners = block(owners, context, owner_mask.to(torch.float32))
        contacts = self.contact_head(owners)[:, :, 0].masked_fill(~owner_mask, -30)
        gathered = torch.gather(owners, 1, muscle_owner[:, :, None].expand(-1, -1, owners.shape[-1]))
        muscles = torch.sigmoid(self.muscle_head(torch.cat((gathered, muscle_meta.float()), dim=-1))[:, :, 0])
        muscles = muscles * muscle_mask.to(muscles.dtype)
        return FeedbackOutput(muscles, contacts, self.body_head(context)[:, 0])
