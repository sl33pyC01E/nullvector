from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..world_action_sparse_v5.contract import ModelConfig
from ..world_action_sparse_v5.model import SparseActionDiT
from .contract import CHECKPOINT_FORMAT, source_sha256


class SparseWorldActionV6Runtime:
    def __init__(self, model, device, report, mean, std):
        self.model = model
        self.device = device
        self.report = report
        self.mean = mean
        self.std = std

    @classmethod
    def from_checkpoint(cls, path: Path, *, device="cuda"):
        target = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        if payload.get("format") != CHECKPOINT_FORMAT or payload.get("source_sha256") != source_sha256():
            raise ValueError("sparse v6 world action checkpoint provenance drifted")
        model = SparseActionDiT(ModelConfig(**payload["model_config"]))
        model.load_state_dict(payload["ema_state"])
        model.to(target).eval()
        mean = torch.as_tensor(payload["latent_mean"], device=target).view(1, -1, 1, 1)
        std = torch.as_tensor(payload["latent_std"], device=target).view(1, -1, 1, 1)
        return cls(model, target, payload.get("report", {}), mean, std)

    def predict_latent(self, current, *, action, control, state, return_gate=False):
        current = torch.as_tensor(current, dtype=torch.float32, device=self.device)
        count = len(current)
        action_array = np.asarray(action)
        action_tensor = torch.full((count,), int(action_array), dtype=torch.long, device=self.device) if action_array.ndim == 0 else torch.as_tensor(action_array, dtype=torch.long, device=self.device).reshape(count)
        control_tensor = torch.as_tensor(control, dtype=torch.float32, device=self.device).reshape(count, 4)
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).reshape(count, 64)
        normalized = (current - self.mean) / self.std
        with torch.inference_mode():
            edited, gate, _, _ = self.model.edit(normalized, torch.zeros(count, device=self.device), action_tensor, control_tensor, state_tensor)
        result = edited * self.std + self.mean
        return (result, gate) if return_gate else result
