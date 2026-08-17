from __future__ import annotations

import torch
from torch import Tensor, nn

from .contract import ACTIONS, GLOBAL_FEATURES, MACRO_CHANNELS, MEMBER_FEATURES, MEMBERS, ModelConfig, PATCH, TIMELINE, TIMELINE_FEATURES


class ResidualMLP(nn.Module):
    def __init__(self, width: int):
        super().__init__(); self.norm = nn.LayerNorm(width); self.body = nn.Sequential(nn.Linear(width, width * 2), nn.SiLU(), nn.Linear(width * 2, width))

    def forward(self, value: Tensor) -> Tensor:
        return value + self.body(self.norm(value))


class MacroBlock(nn.Module):
    def __init__(self, width: int, shared: int, dilation: int):
        super().__init__(); self.norm = nn.GroupNorm(8, width); self.condition = nn.Linear(shared, width * 2); self.conv1 = nn.Conv2d(width, width, 3, padding=dilation, dilation=dilation); self.conv2 = nn.Conv2d(width, width, 3, padding=1)

    def forward(self, value: Tensor, shared: Tensor) -> Tensor:
        scale, shift = self.condition(shared).chunk(2, -1); hidden = self.norm(value) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        return value + self.conv2(torch.nn.functional.silu(self.conv1(torch.nn.functional.silu(hidden))))


class MobileCoordinatorStudent(nn.Module):
    """One shared causal graph distilling macro, colony, society and horizon teachers."""

    def __init__(self, config: ModelConfig = ModelConfig()):
        super().__init__(); self.config = config
        summary_features = MACRO_CHANNELS * 4 + GLOBAL_FEATURES * 2 + MEMBER_FEATURES * 2 + 64 + TIMELINE_FEATURES * 2
        self.shared = nn.Sequential(nn.Linear(summary_features, config.shared_width), nn.LayerNorm(config.shared_width), nn.SiLU(), ResidualMLP(config.shared_width), ResidualMLP(config.shared_width))
        self.macro_stem = nn.Conv2d(MACRO_CHANNELS * 2, config.macro_width, 3, padding=1)
        self.macro_blocks = nn.ModuleList(MacroBlock(config.macro_width, config.shared_width, (1, 2, 4, 8)[index % 4]) for index in range(config.macro_blocks))
        self.macro_out = nn.Sequential(nn.GroupNorm(8, config.macro_width), nn.SiLU(), nn.Conv2d(config.macro_width, MACRO_CHANNELS, 1))
        self.macro_global = nn.Linear(config.shared_width, GLOBAL_FEATURES)
        self.member_input = nn.Linear(MEMBER_FEATURES, config.member_width); self.member_condition = nn.Linear(config.shared_width, config.member_width)
        self.member_body = nn.Sequential(ResidualMLP(config.member_width), ResidualMLP(config.member_width)); self.member_role = nn.Linear(config.member_width, 6); self.member_action = nn.Linear(config.member_width, 3)
        self.society_activity = nn.Linear(config.shared_width, 16); self.society_labor = nn.Linear(config.shared_width, 6); self.society_diplomacy = nn.Linear(config.shared_width, 3); self.society_project = nn.Linear(config.shared_width, 9)
        self.timeline_state = nn.Linear(config.shared_width, TIMELINE_FEATURES); self.timeline_event = nn.Linear(config.shared_width, 10); self.timeline_confidence = nn.Linear(config.shared_width, 1)
        self.action = nn.Embedding(ACTIONS, config.shared_width); self.counter_body = ResidualMLP(config.shared_width); self.counter_state = nn.Linear(config.shared_width, TIMELINE_FEATURES); self.counter_value = nn.Linear(config.shared_width, 2)

    @staticmethod
    def _masked_summary(features: Tensor, mask: Tensor) -> Tensor:
        valid = mask.float().unsqueeze(-1); count = valid.sum(1).clamp_min(1); mean = (features * valid).sum(1) / count
        maximum = features.masked_fill(~mask.unsqueeze(-1), -1e4).max(1).values; maximum = torch.where(mask.any(1, keepdim=True), maximum, torch.zeros_like(maximum))
        return torch.cat((mean, maximum), -1)

    def forward(self, current: Tensor, previous: Tensor, global_state: Tensor, previous_global: Tensor,
                members: Tensor, member_mask: Tensor, society: Tensor, sequence: Tensor):
        if current.shape[1:] != (MACRO_CHANNELS, PATCH, PATCH) or members.shape[1:] != (MEMBERS, MEMBER_FEATURES) or sequence.shape[1:] != (TIMELINE, TIMELINE_FEATURES):
            raise ValueError("mobile coordinator input contract drifted")
        macro_mean = current.mean((2, 3)); macro_std = current.std((2, 3), unbiased=False); delta = current - previous
        macro_summary = torch.cat((macro_mean, macro_std, delta.mean((2, 3)), delta.std((2, 3), unbiased=False)), -1)
        summary = torch.cat((macro_summary, global_state, previous_global, self._masked_summary(members, member_mask), society, sequence.mean(1), sequence[:, -1]), -1)
        shared = self.shared(summary)
        hidden = self.macro_stem(torch.cat((current, delta), 1))
        for block in self.macro_blocks: hidden = block(hidden, shared)
        macro_next = (current + .15 * torch.tanh(self.macro_out(hidden))).clamp(0, 1); macro_global = (global_state + .15 * torch.tanh(self.macro_global(shared))).clamp(0, 1)
        member_hidden = self.member_body(self.member_input(members) + self.member_condition(shared)[:, None]); role = self.member_role(member_hidden); actions = torch.sigmoid(self.member_action(member_hidden))
        timeline_state = (sequence[:, -1] + .2 * torch.tanh(self.timeline_state(shared))).clamp(0, 1)
        action_hidden = self.counter_body(shared[:, None] + self.action.weight[None]); counter_state = (sequence[:, -1, None] + .3 * torch.tanh(self.counter_state(action_hidden))).clamp(0, 1); counter_value = torch.sigmoid(self.counter_value(action_hidden))
        return (macro_next, macro_global, role, actions, self.society_activity(shared), self.society_labor(shared),
                self.society_diplomacy(shared), self.society_project(shared), timeline_state,
                self.timeline_event(shared), torch.sigmoid(self.timeline_confidence(shared)).squeeze(-1),
                counter_state, counter_value[..., 0], counter_value[..., 1])
