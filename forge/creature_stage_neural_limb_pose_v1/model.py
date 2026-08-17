from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .contract import CONTEXT_FEATURES, MAX_NODES, NODE_FEATURES, ModelConfig


@dataclass(slots=True)
class LimbPoseOutput:
    pose: Tensor
    confidence: Tensor


class PoseBlock(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(width)
        self.context = nn.Linear(width, width * 2)
        self.net = nn.Sequential(
            nn.Linear(width, width * 3), nn.SiLU(), nn.Dropout(dropout), nn.Linear(width * 3, width),
        )

    def forward(self, value: Tensor, context: Tensor, mask: Tensor) -> Tensor:
        normalized = self.norm1(value)
        attended, _ = self.attention(normalized, normalized, normalized, key_padding_mask=~mask, need_weights=False)
        value = value + attended
        scale, shift = self.context(context).chunk(2, dim=-1)
        conditioned = self.norm2(value) * (1 + scale[:, None]) + shift[:, None]
        return (value + self.net(conditioned)) * mask[:, :, None].to(value.dtype)


class NeuralLimbPose(nn.Module):
    """Conditioned neural inverse-muscle field for arbitrary appendage chains."""

    def __init__(self, config: ModelConfig = ModelConfig()) -> None:
        super().__init__()
        self.config = config
        self.nodes = nn.Sequential(nn.Linear(NODE_FEATURES, config.width), nn.SiLU(), nn.Linear(config.width, config.width))
        self.context = nn.Sequential(nn.Linear(CONTEXT_FEATURES, config.width), nn.SiLU(), nn.Linear(config.width, config.width))
        self.blocks = nn.ModuleList(PoseBlock(config.width, config.heads, config.dropout) for _ in range(config.depth))
        self.pose_head = nn.Sequential(nn.LayerNorm(config.width), nn.Linear(config.width, config.width), nn.SiLU(), nn.Linear(config.width, 2))
        self.confidence_head = nn.Sequential(nn.LayerNorm(config.width), nn.Linear(config.width, 1), nn.Sigmoid())

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, nodes: Tensor, context: Tensor, mask: Tensor) -> LimbPoseOutput:
        batch = nodes.shape[0]
        if (
            nodes.shape != (batch, MAX_NODES, NODE_FEATURES)
            or context.shape != (batch, CONTEXT_FEATURES)
            or mask.shape != (batch, MAX_NODES)
            or mask.dtype is not torch.bool
            or not bool(mask[:, :2].all())
            or not bool(torch.isfinite(nodes).all())
            or not bool(torch.isfinite(context).all())
        ):
            raise ValueError("neural limb pose input drifted")
        value = self.nodes(nodes.float()) * mask[:, :, None].to(torch.float32)
        condition = self.context(context.float())
        for block in self.blocks:
            value = block(value, condition, mask)
        # Physical boundary prior: the root-to-hand chord and cumulative bone
        # fraction are exact inputs. The network learns only the muscle-driven
        # curvature around that chord, so it cannot waste capacity relearning
        # endpoint interpolation or invent a disconnected limb.
        fraction = nodes[:, :, 4:5].float()
        chord = context[:, None, 0:2].float()
        base = fraction * chord
        interior = mask[:, :, None] & (fraction > 0) & (nodes[:, :, 7:8] < .5)
        residual = torch.tanh(self.pose_head(value)) * .32
        pose = (base + residual * interior.to(torch.float32)) * mask[:, :, None].to(torch.float32)
        confidence = self.confidence_head(condition)[:, 0]
        return LimbPoseOutput(pose, confidence)
