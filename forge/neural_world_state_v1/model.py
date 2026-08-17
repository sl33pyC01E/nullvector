from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .contract import CITY_CLASSES, CONDITION_NAMES, CONTINUOUS_NAMES, TERRAIN_CLASSES, WorldStateModelConfig


@dataclass(frozen=True, slots=True)
class WorldStateOutput:
    terrain: Tensor
    city: Tensor
    continuous: Tensor
    condition: Tensor
    mean: Tensor
    logvar: Tensor
    spatial: Tensor
    global_state: Tensor


class NeuralWorldStateVAE(nn.Module):
    def __init__(self, config: WorldStateModelConfig = WorldStateModelConfig()) -> None:
        super().__init__(); self.config = config; width = config.width
        self.terrain_embedding = nn.Embedding(TERRAIN_CLASSES, 16); self.city_embedding = nn.Embedding(CITY_CLASSES, 16)
        self.input = nn.Sequential(nn.Conv2d(32 + len(CONTINUOUS_NAMES), width, 3, padding=1), nn.SiLU(), nn.Conv2d(width, width, 3, padding=1), nn.SiLU())
        self.down1 = nn.Sequential(nn.Conv2d(width, width * 2, 4, 2, 1), nn.GroupNorm(8, width * 2), nn.SiLU(), nn.Conv2d(width * 2, width * 2, 3, padding=1), nn.SiLU())
        self.down2 = nn.Sequential(nn.Conv2d(width * 2, width * 3, 4, 2, 1), nn.GroupNorm(8, width * 3), nn.SiLU())
        self.posterior = nn.Conv2d(width * 3, config.latent_channels * 2, 1)
        self.condition_encoder = nn.Sequential(nn.Linear(len(CONDITION_NAMES), 128), nn.SiLU(), nn.Linear(128, config.global_features))
        self.global_encoder = nn.Sequential(nn.Linear(width * 3 + config.global_features, 128), nn.SiLU(), nn.Linear(128, config.global_features))
        self.latent_in = nn.Conv2d(config.latent_channels, width * 3, 1)
        self.up1 = nn.Sequential(nn.Conv2d(width * 3, width * 2, 3, padding=1), nn.GroupNorm(8, width * 2), nn.SiLU())
        self.up2 = nn.Sequential(nn.Conv2d(width * 2, width, 3, padding=1), nn.GroupNorm(8, width), nn.SiLU())
        self.terrain_head = nn.Conv2d(width, TERRAIN_CLASSES, 1); self.city_head = nn.Conv2d(width, CITY_CLASSES, 1); self.continuous_head = nn.Conv2d(width, len(CONTINUOUS_NAMES), 1)
        self.condition_head = nn.Sequential(nn.Linear(config.global_features, 128), nn.SiLU(), nn.Linear(128, len(CONDITION_NAMES)))

    def encode(self, terrain: Tensor, city: Tensor, continuous: Tensor, condition: Tensor, *, sample: bool = True) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        embedded = torch.cat((self.terrain_embedding(terrain).permute(0, 3, 1, 2), self.city_embedding(city).permute(0, 3, 1, 2), continuous.float()), 1); hidden = self.down2(self.down1(self.input(embedded))); mean, logvar = self.posterior(hidden).chunk(2, 1); logvar = logvar.clamp(-8, 5); spatial = mean + torch.randn_like(mean) * torch.exp(.5 * logvar) if sample else mean; context = self.condition_encoder(condition.float()); global_state = self.global_encoder(torch.cat((hidden.mean((2, 3)), context), 1)); return spatial, global_state, mean, logvar

    def decode(self, spatial: Tensor, global_state: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        hidden = self.latent_in(spatial); hidden = F.interpolate(hidden, scale_factor=2, mode="nearest"); hidden = self.up1(hidden); hidden = F.interpolate(hidden, scale_factor=2, mode="nearest"); hidden = self.up2(hidden); return self.terrain_head(hidden), self.city_head(hidden), torch.sigmoid(self.continuous_head(hidden)), self.condition_head(global_state)

    def forward(self, terrain: Tensor, city: Tensor, continuous: Tensor, condition: Tensor, *, sample: bool = True) -> WorldStateOutput:
        spatial, global_state, mean, logvar = self.encode(terrain, city, continuous, condition, sample=sample); terrain_out, city_out, continuous_out, condition_out = self.decode(spatial, global_state); return WorldStateOutput(terrain_out, city_out, continuous_out, condition_out, mean, logvar, spatial, global_state)


def build_model(config: WorldStateModelConfig, *, seed: int = 0x574F524C44564145) -> NeuralWorldStateVAE:
    previous = torch.get_rng_state()
    try: torch.manual_seed(seed); return NeuralWorldStateVAE(config)
    finally: torch.set_rng_state(previous)
