from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn

from .contract import GLOBAL_FEATURES, PATCH_SIZE, STATE_CHANNELS, ModelConfig


class MacroBlock(nn.Module):
    def __init__(self, width: int, global_width: int, dilation: int):
        super().__init__()
        self.norm = nn.GroupNorm(8, width)
        self.conv1 = nn.Conv2d(width, width, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv2d(width, width, 3, padding=1)
        self.condition = nn.Linear(global_width, width * 2)
        self.act = nn.SiLU()

    def forward(self, value, condition):
        scale, shift = self.condition(condition).chunk(2, -1)
        hidden = self.norm(value) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        return value + self.conv2(self.act(self.conv1(self.act(hidden))))


class NeuralMacroPatchDynamics(nn.Module):
    def __init__(self, config: ModelConfig = ModelConfig()):
        super().__init__()
        self.config = config
        channels = len(STATE_CHANNELS)
        self.global_encoder = nn.Sequential(nn.Linear(GLOBAL_FEATURES * 2, config.global_width), nn.SiLU(), nn.Linear(config.global_width, config.global_width), nn.SiLU())
        self.stem = nn.Conv2d(channels * 2, config.width, 3, padding=1)
        self.blocks = nn.ModuleList(MacroBlock(config.width, config.global_width, (1, 2, 4, 8)[index % 4]) for index in range(config.blocks))
        self.norm = nn.GroupNorm(8, config.width)
        self.delta = nn.Conv2d(config.width, channels, 1)
        self.gate = nn.Conv2d(config.width, channels, 1)
        self.global_head = nn.Sequential(nn.Linear(config.width + config.global_width, config.global_width), nn.SiLU(), nn.Linear(config.global_width, GLOBAL_FEATURES * 2))
        nn.init.zeros_(self.delta.weight); nn.init.zeros_(self.delta.bias)
        nn.init.zeros_(self.gate.weight); nn.init.constant_(self.gate.bias, -3.0)
        nn.init.zeros_(self.global_head[-1].weight); nn.init.zeros_(self.global_head[-1].bias)
        with torch.no_grad(): self.global_head[-1].bias[GLOBAL_FEATURES:].fill_(-2.5)

    def forward(self, current, previous, global_state, previous_global):
        if current.shape[1:] != (len(STATE_CHANNELS), PATCH_SIZE, PATCH_SIZE) or global_state.shape[1:] != (GLOBAL_FEATURES,):
            raise ValueError("macro neural tensor contract drifted")
        condition = self.global_encoder(torch.cat((global_state, global_state - previous_global), -1))
        hidden = self.stem(torch.cat((current, current - previous), 1))
        for block in self.blocks:
            hidden = block(hidden, condition)
        hidden = torch.nn.functional.silu(self.norm(hidden))
        delta = self.delta(hidden); gate_logits = self.gate(hidden); gate = torch.sigmoid(gate_logits)
        pooled = hidden.mean((2, 3)); global_delta, global_gate_logits = self.global_head(torch.cat((pooled, condition), -1)).chunk(2, -1)
        global_gate = torch.sigmoid(global_gate_logits)
        return (current + gate * delta).clamp(0, 1), (global_state + global_gate * global_delta).clamp(0, 1), gate, gate_logits, global_gate, global_gate_logits
