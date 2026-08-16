from __future__ import annotations

import torch
from torch import nn

from ..action_teacher_v1.contract import ACTIONS as ACTION_NAMES
from ..action_teacher_v2.actor import ACTOR_FEATURE_NAMES
from ..action_teacher_v2.contract import ACTOR_FEATURES, ACTOR_FIELD_SHAPE
from ..world_action_sparse_v5.model import SparseActionDiT, SparseBlock, _modulate, spatial_control_fields
from ..world_latent_dit.contract import ACTIONS, CONTROL_FEATURES, LATENT_CHANNELS, LATENT_SIZE, STATE_FEATURES
from .contract import ModelConfig


class CellularTemporalActionDiT(nn.Module):
    """Sparse action editor with visual motion and privileged cellular state.

    Extra adapters and state-transition heads are zero-initialized.  Loading a
    v5 editor therefore preserves its exact latent prediction at step zero while
    the new model initially predicts persistence for actor physiology/anatomy.
    """

    def __init__(self, config: ModelConfig = ModelConfig()):
        super().__init__()
        self.config = config
        tokens = (LATENT_SIZE // config.patch) ** 2
        self.patch = nn.Conv2d(LATENT_CHANNELS, config.width, config.patch, config.patch)
        self.spatial = nn.Conv2d(config.spatial_channels, config.width, config.patch, config.patch)
        self.position = nn.Parameter(torch.randn(1, tokens, config.width) * 0.015)
        self.action = nn.Embedding(ACTIONS, config.width)
        self.control = nn.Linear(CONTROL_FEATURES, config.width)
        self.state = nn.Linear(STATE_FEATURES, config.width)
        self.time = nn.Sequential(nn.Linear(64, config.width), nn.SiLU(), nn.Linear(config.width, config.width))
        self.blocks = nn.ModuleList(SparseBlock(config.width, config.heads) for _ in range(config.layers))
        self.norm = nn.LayerNorm(config.width, elementwise_affine=False)
        self.final_mod = nn.Sequential(nn.SiLU(), nn.Linear(config.width, config.width * 2))
        self.delta_out = nn.Linear(config.width, LATENT_CHANNELS * config.patch * config.patch)
        self.gate_out = nn.Linear(config.width, config.patch * config.patch)

        self.velocity_patch = nn.Conv2d(LATENT_CHANNELS, config.width, config.patch, config.patch)
        self.actor_spatial = nn.Conv2d(config.actor_field_channels, config.width, config.patch, config.patch)
        self.previous_action = nn.Embedding(ACTIONS, config.width)
        self.previous_control = nn.Linear(CONTROL_FEATURES, config.width)
        self.actor_state_condition = nn.Linear(ACTOR_FEATURES, config.width)
        self.actor_state_out = nn.Linear(config.width, ACTOR_FEATURES)
        self.actor_field_out = nn.Linear(config.width, ACTOR_FIELD_SHAPE[0] * config.patch * config.patch)
        for module in (self.velocity_patch, self.actor_spatial, self.previous_action, self.previous_control, self.actor_state_condition, self.actor_state_out, self.actor_field_out):
            nn.init.zeros_(module.weight)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)
        immutable = torch.zeros(ACTOR_FEATURES, dtype=torch.bool)
        for index, name in enumerate(ACTOR_FEATURE_NAMES):
            immutable[index] = name.startswith(("family_", "stage_", "family_mix_", "development_", "ecology_", "diet_"))
        topology_actions = torch.zeros(len(ACTION_NAMES), dtype=torch.bool)
        for name in ("graft_organ", "graft_locomotor", "metamorphosis"):
            topology_actions[ACTION_NAMES.index(name)] = True
        structural_actions = topology_actions.clone()
        for name in ("impact", "scrape", "cut", "beam", "projectile", "intervention"):
            structural_actions[ACTION_NAMES.index(name)] = True
        self.register_buffer("immutable_actor_mask", immutable, persistent=False)
        self.register_buffer("topology_action_lookup", topology_actions, persistent=False)
        self.register_buffer("structural_action_lookup", structural_actions, persistent=False)

    @staticmethod
    def time_embedding(time):
        return SparseActionDiT.time_embedding(time)

    def _unpatch(self, value, channels):
        batch = value.shape[0]
        patch = self.config.patch
        side = LATENT_SIZE // patch
        return value.view(batch, side, side, channels, patch, patch).permute(0, 3, 1, 4, 2, 5).reshape(batch, channels, LATENT_SIZE, LATENT_SIZE)

    def forward(self, current, previous, time, action, control, state, actor_state, actor_field, previous_action, previous_control):
        if actor_field.shape[1:] != ACTOR_FIELD_SHAPE:
            raise ValueError("cellular action actor field drifted")
        condition = self.time(self.time_embedding(time)) + self.action(action) + self.control(control) + self.state(state)
        condition = condition + self.previous_action(previous_action) + self.previous_control(previous_control) + self.actor_state_condition(actor_state)
        field = spatial_control_fields(control, current.shape[-1])
        token = self.patch(current).flatten(2).transpose(1, 2)
        token = token + self.velocity_patch(current - previous).flatten(2).transpose(1, 2)
        token = token + self.spatial(field).flatten(2).transpose(1, 2)
        token = token + self.actor_spatial(actor_field).flatten(2).transpose(1, 2) + self.position
        for block in self.blocks:
            token = block(token, condition)
        shift, scale = self.final_mod(condition).chunk(2, 1)
        token = _modulate(self.norm(token), shift, scale)
        delta = self._unpatch(self.delta_out(token), LATENT_CHANNELS)
        gate_logits = self._unpatch(self.gate_out(token), 1)
        pooled = token.mean(1)
        actor_state_delta = self.actor_state_out(pooled)
        actor_field_delta = self._unpatch(self.actor_field_out(token), ACTOR_FIELD_SHAPE[0])
        return delta, gate_logits, actor_state_delta, actor_field_delta

    def edit(self, current, previous, time, action, control, state, actor_state, actor_field, previous_action, previous_control):
        delta, gate_logits, actor_state_delta, actor_field_delta = self(current, previous, time, action, control, state, actor_state, actor_field, previous_action, previous_control)
        gate = torch.sigmoid(gate_logits)
        topology = self.topology_action_lookup[action].to(actor_state_delta.dtype)
        structural = self.structural_action_lookup[action].to(actor_state_delta.dtype)
        actor_delta_mask = torch.ones_like(actor_state_delta)
        actor_delta_mask[:, self.immutable_actor_mask] = topology[:, None]
        next_actor_state = actor_state + actor_state_delta * actor_delta_mask
        topology_field = topology[:, None, None, None]
        structural_field = structural[:, None, None, None]
        occupied = actor_field[:, :1].clamp(0, 1)
        dynamic_support = torch.maximum(occupied, topology_field)
        structural_support = torch.maximum(occupied * structural_field, topology_field)
        field_delta_mask = torch.zeros_like(actor_field_delta)
        field_delta_mask[:, 1:5] = dynamic_support
        field_delta_mask[:, :1] = structural_support
        field_delta_mask[:, 5:] = structural_support
        next_actor_field = (actor_field + actor_field_delta * field_delta_mask).clamp(0, 1)
        return current + gate * delta, next_actor_state, next_actor_field, gate, delta, gate_logits


def load_v5_latent_editor(model: CellularTemporalActionDiT, parent: SparseActionDiT) -> tuple[str, ...]:
    """Load the exact v5 latent path while leaving new cellular paths neutral."""
    parent_state = parent.state_dict()
    result = model.load_state_dict(parent_state, strict=False)
    if result.unexpected_keys:
        raise ValueError("v5 warm start contained unexpected parameters: " + ",".join(result.unexpected_keys))
    expected_missing = {
        name
        for name in model.state_dict()
        if name.startswith(("velocity_patch.", "actor_spatial.", "previous_action.", "previous_control.", "actor_state_condition.", "actor_state_out.", "actor_field_out."))
    }
    if set(result.missing_keys) != expected_missing:
        raise ValueError("cellular warm-start parameter closure drifted")
    return tuple(sorted(result.missing_keys))
