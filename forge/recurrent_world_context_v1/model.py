from __future__ import annotations

import torch
from torch import Tensor, nn

from .contract import ContextModelConfig


class WorldContextStateAdapter(nn.Module):
    def __init__(self, config: ContextModelConfig = ContextModelConfig()) -> None:
        super().__init__(); self.config = config
        self.network = nn.Sequential(nn.LayerNorm(config.input_features), nn.Linear(config.input_features, config.width), nn.SiLU(), nn.Linear(config.width, config.width), nn.SiLU(), nn.Linear(config.width, config.output_features))

    def forward(self, context: Tensor) -> Tensor:
        if context.ndim != 2 or context.shape[1] != self.config.input_features or not bool(torch.isfinite(context).all()): raise ValueError("World context adapter input drifted.")
        return self.network(context.float())


def build_model(config: ContextModelConfig, *, seed: int = 0x434F4E544558544D) -> WorldContextStateAdapter:
    previous = torch.get_rng_state()
    try: torch.manual_seed(seed); return WorldContextStateAdapter(config)
    finally: torch.set_rng_state(previous)
