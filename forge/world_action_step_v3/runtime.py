from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..world_latent_dit.contract import ModelConfig as BackboneConfig
from ..world_latent_dit.model import ActionDiT
from .contract import CHECKPOINT_FORMAT, ModelConfig, source_sha256


class WorldActionStepRuntime:
    def __init__(self, model, device, report, mean, std):
        self.model = model
        self.device = device
        self.report = report
        self.mean = mean
        self.std = std

    @classmethod
    def from_checkpoint(cls, path: Path, *, device: str = "cuda"):
        target = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        if payload.get("format") != CHECKPOINT_FORMAT or payload.get("source_sha256") != source_sha256():
            raise ValueError("causal world action checkpoint provenance drifted")
        config = ModelConfig(**payload["model_config"])
        model = ActionDiT(BackboneConfig(**config.__dict__ if hasattr(config, "__dict__") else {name: getattr(config, name) for name in ("width", "layers", "heads", "patch")}))
        model.load_state_dict(payload["ema_state"])
        model.to(target).eval()
        mean = torch.as_tensor(payload["latent_mean"], device=target).view(1, -1, 1, 1)
        std = torch.as_tensor(payload["latent_std"], device=target).view(1, -1, 1, 1)
        return cls(model, target, payload.get("report", {}), mean, std)

    def predict_latent(self, current, *, action, control, state):
        current = torch.as_tensor(current, dtype=torch.float32, device=self.device)
        value = (current - self.mean) / self.std
        count = len(value)
        action_array = np.asarray(action)
        action_tensor = torch.full((count,), int(action_array), dtype=torch.long, device=self.device) if action_array.ndim == 0 else torch.as_tensor(action_array, dtype=torch.long, device=self.device).reshape(count)
        control_tensor = torch.as_tensor(control, dtype=torch.float32, device=self.device).reshape(count, 4)
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).reshape(count, 64)
        time = torch.zeros(count, device=self.device)
        with torch.inference_mode():
            delta = self.model(value, time, action_tensor, control_tensor, state_tensor)
        return current + delta * self.std

    def rollout(self, current, *, actions, controls, states):
        value = torch.as_tensor(current, dtype=torch.float32, device=self.device)
        for action, control, state in zip(actions, controls, states):
            value = self.predict_latent(value, action=action, control=control, state=state)
        return value
