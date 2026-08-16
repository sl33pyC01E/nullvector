from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..action_teacher_v2.contract import ACTOR_FEATURES, ACTOR_FIELD_SHAPE
from ..world_latent_dit.contract import LATENT_CHANNELS, LATENT_SIZE
from .contract import CHECKPOINT_FORMAT, ModelConfig, source_sha256
from .model import CellularTemporalActionDiT


class CellularWorldActionRuntime:
    def __init__(self, model, device, payload):
        self.model = model
        self.device = device
        self.report = payload.get("report", {})
        self.latent_mean = torch.as_tensor(payload["latent_mean"], device=device).view(1, LATENT_CHANNELS, 1, 1)
        self.latent_std = torch.as_tensor(payload["latent_std"], device=device).view(1, LATENT_CHANNELS, 1, 1)
        self.actor_mean = torch.as_tensor(payload["actor_mean"], device=device).view(1, ACTOR_FEATURES)
        self.actor_std = torch.as_tensor(payload["actor_std"], device=device).view(1, ACTOR_FEATURES)

    @classmethod
    def from_checkpoint(cls, path: Path, *, device="cuda"):
        target = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        if payload.get("format") != CHECKPOINT_FORMAT or payload.get("source_sha256") != source_sha256() or payload.get("status") != "evaluated":
            raise ValueError("cellular action runtime provenance drifted")
        state = payload.get("runtime_ema_state", payload.get("best_ema_state"))
        if not isinstance(state, dict):
            raise ValueError("cellular action runtime state missing")
        model = CellularTemporalActionDiT(ModelConfig(**payload["model_config"]))
        model.load_state_dict(state)
        model.to(target).eval()
        return cls(model, target, payload)

    def predict(self, current, previous, *, action, control, state, actor_state, actor_field, previous_action, previous_control):
        current = torch.as_tensor(current, dtype=torch.float32, device=self.device).reshape(-1, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE)
        previous = torch.as_tensor(previous, dtype=torch.float32, device=self.device).reshape_as(current)
        count = len(current)
        action = torch.as_tensor(np.broadcast_to(action, (count,)), dtype=torch.long, device=self.device)
        previous_action = torch.as_tensor(np.broadcast_to(previous_action, (count,)), dtype=torch.long, device=self.device)
        control = torch.as_tensor(control, dtype=torch.float32, device=self.device).reshape(count, 4)
        previous_control = torch.as_tensor(previous_control, dtype=torch.float32, device=self.device).reshape(count, 4)
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device).reshape(count, 64)
        actor_state = torch.as_tensor(actor_state, dtype=torch.float32, device=self.device).reshape(count, ACTOR_FEATURES)
        actor_field = torch.as_tensor(actor_field, dtype=torch.float32, device=self.device).reshape(count, *ACTOR_FIELD_SHAPE)
        current_n = (current - self.latent_mean) / self.latent_std
        previous_n = (previous - self.latent_mean) / self.latent_std
        actor_n = (actor_state - self.actor_mean) / self.actor_std
        with torch.inference_mode():
            latent_n, next_actor_n, next_field, gate, _, _ = self.model.edit(current_n, previous_n, torch.zeros(count, device=self.device), action, control, state, actor_n, actor_field, previous_action, previous_control)
        return {
            "latent": latent_n * self.latent_std + self.latent_mean,
            "actor_state": next_actor_n * self.actor_std + self.actor_mean,
            "actor_field": next_field,
            "gate": gate,
        }
