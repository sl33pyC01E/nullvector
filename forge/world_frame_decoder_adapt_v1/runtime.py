from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..world_frame_vae.contract import ModelConfig
from ..world_frame_vae.model import WorldFrameVAE
from .contract import CHECKPOINT_FORMAT, source_sha256, state_sha256


class AdaptedWorldFrameCodec:
    def __init__(self, model: WorldFrameVAE, device: torch.device, report: dict):
        self.model = model
        self.device = device
        self.report = report

    @classmethod
    def from_checkpoint(cls, path: Path, *, device: str = "cuda"):
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        if payload.get("format") != CHECKPOINT_FORMAT or payload.get("source_sha256") != source_sha256():
            raise ValueError("adapted world decoder provenance drifted")
        if state_sha256(payload["state"]) != payload.get("state_sha256"):
            raise ValueError("adapted world decoder state drifted")
        target = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
        model = WorldFrameVAE(ModelConfig(**payload["model_config"]))
        model.load_state_dict(payload["state"], strict=True)
        model.to(target).eval()
        return cls(model, target, payload["report"])

    @torch.inference_mode()
    def encode(self, frame: np.ndarray):
        if frame.shape != (256, 256, 3) or frame.dtype != np.uint8:
            raise ValueError("world frame must be uint8 HWC 256x256 RGB")
        tensor = torch.from_numpy(frame.copy()).permute(2, 0, 1)[None].float().div_(255).to(self.device)
        return self.model.encode(tensor)[0]

    @torch.inference_mode()
    def decode(self, latent):
        result = self.model.decode(torch.as_tensor(latent, device=self.device)).float().cpu()
        return np.clip(result.permute(0, 2, 3, 1).numpy() * 255, 0, 255).astype(np.uint8)
