from __future__ import annotations

import torch
from torch import Tensor, nn

from ..recurrent_action_dit_v2.model import RecurrentActionDiT
from ..world_latent_dit.contract import LATENT_CHANNELS, LATENT_SIZE, ModelConfig as ActionConfig
from ..world_latent_dit.model import DiTBlock
from .contract import ModelConfig


class DesktopWorldMonolith(nn.Module):
    """One action-conditioned transformer over visual, organism and world tokens."""

    def __init__(self, config: ModelConfig = ModelConfig(), action_config: ActionConfig = ActionConfig()):
        super().__init__()
        if config.width != action_config.width:
            raise ValueError("desktop fusion width must match the pretrained Action-DiT")
        self.config = config
        self.action_config = action_config
        width = config.width
        self.visual = RecurrentActionDiT(action_config)
        self.macro_patch = nn.Conv2d(64, width, config.macro_patch, config.macro_patch)
        self.macro_position = nn.Parameter(torch.randn(1, 64, width) * .015)
        self.global_token = nn.Linear(88, width)
        self.member_token = nn.Linear(64, width)
        self.society_token = nn.Linear(64, width)
        self.timeline_token = nn.Linear(64, width)
        self.counter_queries = nn.Parameter(torch.randn(1, 5, width) * .015)
        self.type_embedding = nn.Parameter(torch.randn(7, width) * .015)
        self.fusion = nn.ModuleList(DiTBlock(width, config.heads) for _ in range(config.fusion_layers))
        self.visual_fusion_gate = nn.Parameter(torch.tensor(-4.0))
        patch = config.macro_patch
        self.macro_out = nn.Linear(width, 32 * patch * patch)
        self.global_out = nn.Linear(width, 44)
        self.member_role = nn.Linear(width, 6)
        self.member_action = nn.Linear(width, 3)
        self.society_activity = nn.Linear(width, 16)
        self.society_labor = nn.Linear(width, 6)
        self.society_diplomacy = nn.Linear(width, 3)
        self.society_project = nn.Linear(width, 9)
        self.timeline_state = nn.Linear(width, 64)
        self.timeline_event = nn.Linear(width, 10)
        self.timeline_confidence = nn.Linear(width, 1)
        self.counter_state = nn.Linear(width, 64)
        self.counter_value = nn.Linear(width, 2)

    def load_action_parent(self, state: dict[str, Tensor]) -> None:
        self.visual.load_state_dict(state, strict=True)

    def freeze_action_parent(self) -> None:
        self.visual.requires_grad_(False)

    def _condition(self, action: Tensor, control: Tensor, state: Tensor, actor: Tensor, dtype: torch.dtype) -> Tensor:
        backbone = self.visual.backbone
        time = torch.zeros(len(action), device=action.device, dtype=dtype)
        actor_conditioned_state = state + self.visual.actor(actor.float())
        return (backbone.time(backbone.time_embedding(time)) + backbone.action(action) +
                backbone.control(control) + backbone.state(actor_conditioned_state))

    def _visual_tokens(self, current: Tensor, previous: Tensor, condition: Tensor) -> Tensor:
        backbone = self.visual.backbone
        remembered = current + self.visual.history(current - previous)
        tokens = backbone.patch(remembered).flatten(2).transpose(1, 2) + backbone.position
        for block in backbone.blocks:
            tokens = block(tokens, condition)
        return tokens

    def _decode_visual(self, tokens: Tensor, condition: Tensor) -> Tensor:
        backbone = self.visual.backbone
        shift, scale = backbone.final_mod(condition).chunk(2, 1)
        value = backbone.norm(tokens) * (1 + scale[:, None]) + shift[:, None]
        value = backbone.out(value)
        batch = len(value); patch = self.action_config.patch; side = LATENT_SIZE // patch
        return value.view(batch, side, side, LATENT_CHANNELS, patch, patch).permute(0, 3, 1, 4, 2, 5).reshape(batch, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE)

    def forward(self, current: Tensor, previous: Tensor, action: Tensor, control: Tensor, state: Tensor, actor: Tensor,
                macro: Tensor, previous_macro: Tensor, global_state: Tensor, previous_global: Tensor,
                members: Tensor, member_mask: Tensor, society: Tensor, sequence: Tensor):
        if macro.shape[1:] != (32, 32, 32) or members.shape[1:] != (16, 64) or sequence.shape[1:] != (24, 64):
            raise ValueError("desktop monolith world input contract drifted")
        condition = self._condition(action, control, state, actor, current.dtype)
        visual_parent = self._visual_tokens(current, previous, condition)
        visual = visual_parent
        macro_delta = macro - previous_macro
        macro_tokens = self.macro_patch(torch.cat((macro, macro_delta), 1)).flatten(2).transpose(1, 2) + self.macro_position + self.type_embedding[1]
        global_token = self.global_token(torch.cat((global_state, previous_global), 1))[:, None] + self.type_embedding[2]
        member_tokens = self.member_token(members) + self.type_embedding[3]
        member_tokens = member_tokens * member_mask[:, :, None].float()
        society_token = self.society_token(society)[:, None] + self.type_embedding[4]
        timeline_tokens = self.timeline_token(sequence) + self.type_embedding[5]
        counter_tokens = self.counter_queries.expand(len(current), -1, -1) + self.type_embedding[6]
        tokens = torch.cat((visual, macro_tokens, global_token, member_tokens, society_token, timeline_tokens, counter_tokens), 1)
        for block in self.fusion:
            tokens = block(tokens, condition)
        visual, cursor = tokens[:, :64], 64
        macro_tokens, cursor = tokens[:, cursor:cursor + 64], cursor + 64
        global_token, cursor = tokens[:, cursor], cursor + 1
        member_tokens, cursor = tokens[:, cursor:cursor + 16], cursor + 16
        society_token, cursor = tokens[:, cursor], cursor + 1
        timeline_tokens, cursor = tokens[:, cursor:cursor + 24], cursor + 24
        counter_tokens = tokens[:, cursor:cursor + 5]
        patch = self.config.macro_patch; side = 32 // patch
        macro_value = self.macro_out(macro_tokens).view(len(tokens), side, side, 32, patch, patch).permute(0, 3, 1, 4, 2, 5).reshape(len(tokens), 32, 32, 32)
        next_macro = (macro + .15 * torch.tanh(macro_value)).clamp(0, 1)
        next_global = (global_state + .15 * torch.tanh(self.global_out(global_token))).clamp(0, 1)
        timeline = (sequence[:, -1] + .2 * torch.tanh(self.timeline_state(timeline_tokens[:, -1]))).clamp(0, 1)
        counter_state = (sequence[:, -1, None] + .3 * torch.tanh(self.counter_state(counter_tokens))).clamp(0, 1)
        counter_value = torch.sigmoid(self.counter_value(counter_tokens))
        fused_visual = visual_parent + torch.sigmoid(self.visual_fusion_gate) * (visual - visual_parent)
        return (self._decode_visual(fused_visual, condition), next_macro, next_global,
                self.member_role(member_tokens), torch.sigmoid(self.member_action(member_tokens)),
                self.society_activity(society_token), self.society_labor(society_token),
                self.society_diplomacy(society_token), self.society_project(society_token),
                timeline, self.timeline_event(timeline_tokens[:, -1]),
                torch.sigmoid(self.timeline_confidence(timeline_tokens[:, -1])).squeeze(-1),
                counter_state, counter_value[..., 0], counter_value[..., 1])
