from __future__ import annotations

from pathlib import Path
import torch

from .contract import CHECKPOINT_FORMAT, GLOBAL_FEATURES, STATE_CHANNELS, ModelConfig, source_sha256
from .model import NeuralMacroPatchDynamics


class NeuralMacroPatchRuntime:
    def __init__(self, model, device, spatial_thresholds, global_thresholds):
        self.model, self.device = model, device
        self.spatial_thresholds = torch.tensor(spatial_thresholds, device=device).view(1, -1, 1, 1)
        self.global_thresholds = torch.tensor(global_thresholds, device=device).view(1, -1)

    @classmethod
    def from_checkpoint(cls, path: Path, *, device="cuda"):
        target = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        if payload.get("format") != CHECKPOINT_FORMAT or payload.get("status") != "evaluated" or payload.get("source_sha256") != source_sha256() or not payload.get("report", {}).get("gates", {}).get("all_passed"):
            raise ValueError("macro neural runtime provenance or quality drifted")
        thresholds = payload.get("gate_thresholds", {})
        if len(thresholds.get("spatial", ())) != len(STATE_CHANNELS) or len(thresholds.get("global", ())) != GLOBAL_FEATURES:
            raise ValueError("macro neural runtime gate calibration drifted")
        model = NeuralMacroPatchDynamics(ModelConfig(**payload["model_config"])); model.load_state_dict(payload["ema_state"])
        return cls(model.to(target).eval(), target, thresholds["spatial"], thresholds["global"])

    @torch.inference_mode()
    def step(self, current, previous, global_state, previous_global):
        tensors = [torch.as_tensor(value, dtype=torch.float32, device=self.device).unsqueeze(0) for value in (current, previous, global_state, previous_global)]
        state, global_next, gate, _, global_gate, _ = self.model(*tensors)
        state = torch.where(gate >= self.spatial_thresholds, state, tensors[0])
        global_next = torch.where(global_gate >= self.global_thresholds, global_next, tensors[2])
        return state[0].cpu().numpy(), global_next[0].cpu().numpy(), gate[0].cpu().numpy()
