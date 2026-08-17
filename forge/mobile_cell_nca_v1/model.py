from __future__ import annotations

import torch
from torch import nn

from ..cellular_nca.contract import BOND_CHANNELS, DYNAMIC_CHANNELS, STATIC_CHANNELS
from .contract import MobileCellNCAConfig


class MobileResidualCellBlock(nn.Module):
    def __init__(self, width: int, expansion: int, dilation: int) -> None:
        super().__init__(); hidden = width * expansion
        self.norm = nn.GroupNorm(8, width)
        self.spatial = nn.Conv2d(width, width, 3, padding=dilation, dilation=dilation, groups=width)
        self.inward = nn.Conv2d(width, hidden * 2, 1); self.outward = nn.Conv2d(hidden, width, 1)
        self.scale = nn.Parameter(torch.full((1, width, 1, 1), .08))

    def forward(self, value):
        update = self.spatial(torch.nn.functional.silu(self.norm(value))); content, gate = self.inward(update).chunk(2, 1)
        return value + self.outward(torch.nn.functional.silu(content) * gate.sigmoid()) * self.scale


class MobileCellNCA(nn.Module):
    def __init__(self, config: MobileCellNCAConfig = MobileCellNCAConfig()) -> None:
        super().__init__(); self.config = config; width = config.width
        self.static_encoder = nn.Sequential(nn.Conv2d(STATIC_CHANNELS, width, 1), nn.SiLU(), nn.Conv2d(width, width, 1))
        self.dynamic_encoder = nn.Conv2d(DYNAMIC_CHANNELS + BOND_CHANNELS, width, 3, padding=1)
        dilations = (1, 2, 1, 3, 1, 2, 1, 4)
        self.blocks = nn.ModuleList(MobileResidualCellBlock(width, config.expansion, dilations[index]) for index in range(config.depth))
        self.readout = nn.Sequential(nn.GroupNorm(8, width), nn.SiLU(), nn.Conv2d(width, DYNAMIC_CHANNELS, 1))
        nn.init.normal_(self.readout[-1].weight, std=1e-4); nn.init.zeros_(self.readout[-1].bias)

    def forward(self, static, state, live_bonds):
        if static.ndim != 4 or static.shape[1:] != (STATIC_CHANNELS, 48, 48):
            raise ValueError("Mobile NCA static tensor contract drifted.")
        if state.shape != (len(static), DYNAMIC_CHANNELS, 48, 48):
            raise ValueError("Mobile NCA state tensor contract drifted.")
        if live_bonds.shape != (len(static), BOND_CHANNELS, 48, 48):
            raise ValueError("Mobile NCA bond tensor contract drifted.")
        value = self.static_encoder(static.float()) + self.dynamic_encoder(torch.cat((state.float(), live_bonds.float()), 1))
        for block in self.blocks: value = block(value)
        result = torch.clamp(state.float() + torch.tanh(self.readout(value)) * self.config.max_delta, 0, 1); body = static[:, :1].float(); result[:, :9] *= body; result[:, 11:12] *= body; return result

    @property
    def parameter_count(self): return sum(value.numel() for value in self.parameters())
