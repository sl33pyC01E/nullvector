from __future__ import annotations

import torch
from torch import Tensor, nn

from .contract import BOND_CHANNELS, DYNAMIC_CHANNELS, STATIC_CHANNELS, CellularNCAConfig


class ResidualCellBlock(nn.Module):
    def __init__(self, width: int, expansion: int, dilation: int) -> None:
        super().__init__()
        hidden = width * expansion
        self.norm = nn.GroupNorm(32, width)
        # A full learned spatial mixing kernel is intentional here.  The first
        # 222 MiB prototype left most of the 4090 idle and visibly collapsed
        # disparate tissues into one response.  Full channel mixing lets each
        # block learn distinct circulation/neural/immune/material interactions.
        self.spatial = nn.Conv2d(width, width, 3, padding=dilation, dilation=dilation)
        self.inward = nn.Conv2d(width, hidden * 2, 1)
        self.outward = nn.Conv2d(hidden, width, 1)
        self.scale = nn.Parameter(torch.full((1, width, 1, 1), 0.08))

    def forward(self, value: Tensor) -> Tensor:
        update = self.spatial(torch.nn.functional.silu(self.norm(value)))
        content, gate = self.inward(update).chunk(2, dim=1)
        update = self.outward(torch.nn.functional.silu(content) * gate.sigmoid())
        return value + update * self.scale


class OrganismCellularAutomaton(nn.Module):
    """Organ-conditioned neural update rule over native 48x48 physical cells.

    Static anatomy is immutable. Dynamic state and the eight directed live-bond
    planes are the only evolving inputs. The returned state is bounded and the
    model cannot create body cells outside the anatomy mask; leaked surface
    fluid and biomass remain free to diffuse through the surrounding plane.
    """

    def __init__(self, config: CellularNCAConfig = CellularNCAConfig()) -> None:
        super().__init__()
        self.config = config
        width = config.width
        self.static_encoder = nn.Sequential(
            nn.Conv2d(STATIC_CHANNELS, width, 1), nn.SiLU(), nn.Conv2d(width, width, 1),
        )
        self.dynamic_encoder = nn.Conv2d(DYNAMIC_CHANNELS + BOND_CHANNELS, width, 3, padding=1)
        dilations = (1, 2, 1, 3, 1, 2, 1, 4, 1, 2, 1, 3, 1, 2, 1, 4)
        self.blocks = nn.ModuleList(
            ResidualCellBlock(width, config.expansion, dilations[index]) for index in range(config.depth)
        )
        self.readout = nn.Sequential(nn.GroupNorm(32, width), nn.SiLU(), nn.Conv2d(width, DYNAMIC_CHANNELS, 1))
        nn.init.normal_(self.readout[-1].weight, std=1e-4)
        nn.init.zeros_(self.readout[-1].bias)

    def forward(self, static: Tensor, state: Tensor, live_bonds: Tensor) -> Tensor:
        if static.ndim != 4 or static.shape[1] != STATIC_CHANNELS or state.shape != (len(static), DYNAMIC_CHANNELS, 48, 48) or live_bonds.shape != (len(static), BOND_CHANNELS, 48, 48):
            raise ValueError("Cellular NCA tensor contract drifted.")
        value = self.static_encoder(static.float()) + self.dynamic_encoder(torch.cat((state.float(), live_bonds.float()), dim=1))
        for block in self.blocks:
            value = block(value)
        delta = torch.tanh(self.readout(value)) * self.config.max_delta
        next_state = torch.clamp(state.float() + delta, 0.0, 1.0)
        body = static[:, :1].float()
        # Internal physiology never materializes outside the immutable chassis.
        next_state[:, :9] *= body
        next_state[:, 11:12] *= body
        return next_state


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
