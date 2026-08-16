from __future__ import annotations

from pathlib import Path
import torch

from .contract import CHECKPOINT_FORMAT, ModelConfig, source_sha256
from .model import NeuralMacroPatchDynamics


class NeuralMacroPatchRuntime:
    def __init__(self, model, device): self.model, self.device = model, device

    @classmethod
    def from_checkpoint(cls, path: Path, *, device="cuda"):
        target = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        if payload.get("format") != CHECKPOINT_FORMAT or payload.get("status") != "evaluated" or payload.get("source_sha256") != source_sha256() or not payload.get("report", {}).get("gates", {}).get("all_passed"):
            raise ValueError("macro neural runtime provenance or quality drifted")
        model = NeuralMacroPatchDynamics(ModelConfig(**payload["model_config"])); model.load_state_dict(payload["ema_state"])
        return cls(model.to(target).eval(), target)

    @torch.inference_mode()
    def step(self, current, previous, global_state, previous_global):
        tensors = [torch.as_tensor(value, dtype=torch.float32, device=self.device).unsqueeze(0) for value in (current, previous, global_state, previous_global)]
        state, global_next, gate, *_ = self.model(*tensors)
        return state[0].cpu().numpy(), global_next[0].cpu().numpy(), gate[0].cpu().numpy()
