from __future__ import annotations

import torch
from torch import Tensor, nn

from ..neural_world_state_v1.contract import CITY_CLASSES, CONDITION_NAMES, CONTINUOUS_NAMES, TERRAIN_CLASSES
from ..recurrent_world_student_v5.model import PerceptionRecurrentWorldStudent
from ..world_latent_dit.contract import ModelConfig as RecurrentConfig
from .contract import DirectContextConfig


class DirectWorldContextEncoder(nn.Module):
    """Distilled structured observation encoder used directly by the action model."""

    def __init__(self, config: DirectContextConfig = DirectContextConfig()) -> None:
        super().__init__()
        self.config = config
        embed = config.embedding_features
        width = config.width
        self.terrain = nn.Embedding(TERRAIN_CLASSES, embed)
        self.city = nn.Embedding(CITY_CLASSES, embed)
        self.spatial = nn.Sequential(
            nn.Conv2d(embed * 2 + len(CONTINUOUS_NAMES), width, 3, padding=1), nn.SiLU(),
            nn.Conv2d(width, width * 2, 4, 2, 1), nn.GroupNorm(8, width * 2), nn.SiLU(),
            nn.Conv2d(width * 2, width * 2, 4, 2, 1), nn.GroupNorm(8, width * 2), nn.SiLU(),
            nn.Conv2d(width * 2, width * 2, 3, padding=1), nn.SiLU(),
        )
        self.condition = nn.Sequential(nn.Linear(len(CONDITION_NAMES), width * 2), nn.SiLU())
        self.output = nn.Sequential(
            nn.LayerNorm(width * 4), nn.Linear(width * 4, width * 2), nn.SiLU(),
            nn.Linear(width * 2, config.output_features),
        )

    def forward(self, terrain: Tensor, city: Tensor, continuous: Tensor, condition: Tensor) -> Tensor:
        if terrain.ndim != 3 or terrain.shape[-2:] != (32, 32) or city.shape != terrain.shape:
            raise ValueError("Direct world-context categorical shape drifted.")
        if continuous.shape != (len(terrain), len(CONTINUOUS_NAMES), 32, 32):
            raise ValueError("Direct world-context continuous shape drifted.")
        if condition.shape != (len(terrain), len(CONDITION_NAMES)):
            raise ValueError("Direct world-context condition shape drifted.")
        embedded = torch.cat((
            self.terrain(terrain.long()).permute(0, 3, 1, 2),
            self.city(city.long()).permute(0, 3, 1, 2),
            continuous.float(),
        ), 1)
        spatial = self.spatial(embedded).mean((2, 3))
        return self.output(torch.cat((spatial, self.condition(condition.float())), 1))


class FusedStructuredActionModel(nn.Module):
    """Single action/world model: structured state in, causal latent edits out."""

    def __init__(self, context_config: DirectContextConfig, recurrent_config: RecurrentConfig) -> None:
        super().__init__()
        self.context = DirectWorldContextEncoder(context_config)
        self.recurrent = PerceptionRecurrentWorldStudent(recurrent_config)

    def observe(self, terrain: Tensor, city: Tensor, continuous: Tensor, condition: Tensor) -> Tensor:
        return self.context(terrain, city, continuous, condition)

    def action(self, current: Tensor, previous: Tensor, action: Tensor, control: Tensor,
               terrain: Tensor, city: Tensor, continuous: Tensor, condition: Tensor,
               actor_state: Tensor, visibility: Tensor, memory: Tensor):
        state = self.observe(terrain, city, continuous, condition)
        delta, gate = self.recurrent.gated_action(
            current, previous, action, control, state, actor_state, visibility, memory
        )
        return delta, gate, state


def build_encoder(config: DirectContextConfig, *, seed: int = 0x4D4F4E4F454E434F) -> DirectWorldContextEncoder:
    previous = torch.get_rng_state()
    try:
        torch.manual_seed(seed)
        return DirectWorldContextEncoder(config)
    finally:
        torch.set_rng_state(previous)
