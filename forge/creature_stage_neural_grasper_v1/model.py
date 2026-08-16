from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn

from .contract import GLOBAL_FEATURES, MAX_APPENDAGES, OWNER_FEATURES, TARGET_FEATURES, TARGET_TYPES, ModelConfig


@dataclass(slots=True)
class GrasperOutput:
    appendage_logits: torch.Tensor
    engage_logit: torch.Tensor
    reach: torch.Tensor
    force: torch.Tensor
    type_logits: torch.Tensor
    brace: torch.Tensor
    release_logit: torch.Tensor
    throw_impulse: torch.Tensor


class Block(nn.Module):
    def __init__(self, width: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.film = nn.Linear(width, width * 2)
        self.net = nn.Sequential(nn.Linear(width, width * 3), nn.SiLU(), nn.Dropout(dropout), nn.Linear(width * 3, width))

    def forward(self, owner, context, mask):
        scale, shift = self.film(context).chunk(2, -1)
        value = self.norm(owner) * (1 + scale[:, None]) + shift[:, None]
        return (owner + self.net(value)) * mask[:, :, None]


class NeuralGrasperController(nn.Module):
    def __init__(self, config: ModelConfig = ModelConfig()):
        super().__init__(); self.config = config; width = config.width
        self.owner = nn.Sequential(nn.Linear(OWNER_FEATURES, width), nn.SiLU(), nn.Linear(width, width))
        self.context = nn.Sequential(nn.Linear(TARGET_FEATURES + GLOBAL_FEATURES, width), nn.SiLU(), nn.Linear(width, width))
        self.blocks = nn.ModuleList(Block(width, config.dropout) for _ in range(config.depth))
        self.select = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))
        self.command = nn.Sequential(nn.Linear(width * 2, width * 2), nn.SiLU(), nn.Linear(width * 2, 1 + 2 + 1 + len(TARGET_TYPES) + 1 + 1 + 2))

    def forward(self, owner_meta, owner_mask, target, global_state):
        batch = owner_meta.shape[0]
        if owner_meta.shape != (batch, MAX_APPENDAGES, OWNER_FEATURES) or owner_mask.shape != (batch, MAX_APPENDAGES) or owner_mask.dtype is not torch.bool or target.shape != (batch, TARGET_FEATURES) or global_state.shape != (batch, GLOBAL_FEATURES):
            raise ValueError("neural grasper input contract drifted")
        context = self.context(torch.cat((target.float(), global_state.float()), -1))
        value = self.owner(owner_meta.float()) * owner_mask[:, :, None]
        for block in self.blocks:
            value = block(value, context, owner_mask)
        logits = self.select(value)[:, :, 0].masked_fill(~owner_mask, -30)
        weights = torch.softmax(logits, -1)
        selected = (value * weights[:, :, None]).sum(1)
        command = self.command(torch.cat((selected, context), -1))
        engage, reach, force, type_logits, brace, release, throw_impulse = torch.split(command, (1, 2, 1, len(TARGET_TYPES), 1, 1, 2), -1)
        return GrasperOutput(
            logits, engage[:, 0], torch.tanh(reach), torch.sigmoid(force[:, 0]),
            type_logits, torch.sigmoid(brace[:, 0]), release[:, 0], torch.tanh(throw_impulse),
        )

    @property
    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())
