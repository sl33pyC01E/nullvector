from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ..map_topology_neural.contract import GLOBAL_CONDITION_NAMES, POINT_CHANNELS
from ..maps.model import THEMES
from .contract import CODEBOOK_SIZE, MASK_TOKEN, MaskedPriorConfig


class PriorResidual(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        groups = min(8, width)
        while width % groups:
            groups -= 1
        self.norm1 = nn.GroupNorm(groups, width)
        self.conv1 = nn.Conv2d(width, width, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, width)
        self.conv2 = nn.Conv2d(width, width, 3, padding=1)

    def forward(self, value: Tensor, valid: Tensor) -> Tensor:
        hidden = self.conv1(F.silu(self.norm1(value)))
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return (value + hidden) * valid


class MaskedTopologyPrior(nn.Module):
    """Compact spatial masked-token model. It emits raw latent proposals only."""

    def __init__(self, config: MaskedPriorConfig) -> None:
        super().__init__()
        self.config = config
        width = config.width
        self.token_embedding = nn.Embedding(CODEBOOK_SIZE + 1, width)
        self.point_projection = nn.Conv2d(len(POINT_CHANNELS), width, 1)
        self.coordinate_projection = nn.Conv2d(2, width, 1)
        self.theme_embedding = nn.Embedding(len(THEMES), width)
        self.global_projection = nn.Linear(len(GLOBAL_CONDITION_NAMES), width)
        self.mask_projection = nn.Linear(1, width)
        self.input_conv = nn.Conv2d(width, width, 3, padding=1)
        self.blocks = nn.ModuleList(PriorResidual(width) for _ in range(config.residual_depth))
        self.output_norm = nn.GroupNorm(min(8, width), width)
        self.output = nn.Conv2d(width, CODEBOOK_SIZE, 1)

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        tokens = batch["tokens"]
        valid = batch["valid_mask"]
        points = batch["point_conditions"]
        global_conditions = batch["global_conditions"]
        themes = batch["theme_index"]
        mask_fraction = batch["mask_fraction"]
        if tokens.ndim != 3 or tokens.dtype != torch.long:
            raise ValueError("Masked-prior tokens must be int64 B,H,W.")
        batch_size, height, width = tokens.shape
        if bool(((tokens < 0) | (tokens > MASK_TOKEN)).any()):
            raise ValueError("Masked-prior tokens exceed the codebook plus mask vocabulary.")
        if valid.shape != (batch_size, 1, height, width) or valid.dtype != torch.bool:
            raise ValueError("Masked-prior valid mask shape/dtype drifted.")
        if points.shape != (batch_size, len(POINT_CHANNELS), height, width):
            raise ValueError("Masked-prior point conditioning shape drifted.")
        if global_conditions.shape != (batch_size, len(GLOBAL_CONDITION_NAMES)):
            raise ValueError("Masked-prior global conditioning shape drifted.")
        if themes.shape != (batch_size,) or mask_fraction.shape != (batch_size, 1):
            raise ValueError("Masked-prior scalar conditioning shape drifted.")
        valid_float = valid.to(torch.float32)
        y = torch.linspace(-1.0, 1.0, height, dtype=torch.float32, device=tokens.device)
        x = torch.linspace(-1.0, 1.0, width, dtype=torch.float32, device=tokens.device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        coordinates = torch.stack((xx, yy), dim=0).expand(batch_size, -1, -1, -1)
        hidden = self.token_embedding(tokens).permute(0, 3, 1, 2)
        hidden = hidden + self.point_projection(points.float()) + self.coordinate_projection(coordinates)
        condition = (
            self.theme_embedding(themes.long())
            + self.global_projection(global_conditions.float())
            + self.mask_projection(mask_fraction.float())
        )
        hidden = self.input_conv(F.silu(hidden + condition[:, :, None, None])) * valid_float
        for block in self.blocks:
            hidden = block(hidden, valid_float)
        return self.output(F.silu(self.output_norm(hidden)))


def build_prior(config: MaskedPriorConfig, *, init_seed: int | None = None) -> MaskedTopologyPrior:
    seed = config.seed if init_seed is None else int(init_seed)
    if not 0 <= seed < 1 << 63:
        raise ValueError("Masked-prior initialization seed must be unsigned 63-bit.")
    previous = torch.get_rng_state()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    try:
        torch.set_rng_state(generator.get_state())
        model = MaskedTopologyPrior(config)
    finally:
        torch.set_rng_state(previous)
    return model.to(device="cpu", dtype=torch.float32)


def masked_token_loss(logits: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
    if logits.ndim != 4 or logits.shape[1] != CODEBOOK_SIZE:
        raise ValueError("Masked-prior logits have the wrong vocabulary.")
    if targets.shape != logits.shape[:1] + logits.shape[2:] or mask.shape != targets.shape:
        raise ValueError("Masked-prior loss shapes disagree.")
    if mask.dtype != torch.bool or not bool(mask.any()):
        raise ValueError("Masked-prior loss requires at least one masked target.")
    per_cell = F.cross_entropy(logits, targets.long(), reduction="none")
    return per_cell[mask].mean()

