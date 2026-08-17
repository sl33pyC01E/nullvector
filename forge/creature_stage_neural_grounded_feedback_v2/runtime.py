from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..creature_stage_neural_motion.training import _state_sha256
from .contract import CHECKPOINT_FORMAT, ModelConfig, source_sha256
from .dataset import encode_live
from .model import NeuralGroundedFeedback


class NeuralGroundedFeedbackRuntime:
    """Causal live-state policy. Physics remains authoritative."""

    def __init__(self, model: NeuralGroundedFeedback, device: torch.device) -> None:
        self.model = model.eval()
        self.device = device

    @classmethod
    def from_checkpoint(cls, path: Path, *, device: str = "cuda") -> "NeuralGroundedFeedbackRuntime":
        target = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        if payload.get("format") != CHECKPOINT_FORMAT or payload.get("source_sha256") != source_sha256():
            raise ValueError("grounded feedback checkpoint provenance drifted")
        state = payload.get("model_state")
        if not isinstance(state, dict) or _state_sha256(state) != payload.get("model_state_sha256"):
            raise ValueError("grounded feedback state drifted")
        if not payload.get("report", {}).get("gates", {}).get("all_passed"):
            raise ValueError("grounded feedback checkpoint failed quality")
        model = NeuralGroundedFeedback(ModelConfig(**payload["model_config"]))
        model.load_state_dict(state, strict=True)
        return cls(model.to(target), target)

    @torch.inference_mode()
    def predict(self, organism, nodes_local: np.ndarray, node_velocity: np.ndarray,
                previous_contact: np.ndarray, phase: float, body_velocity: float
                ) -> tuple[np.ndarray, np.ndarray, float]:
        encoded = encode_live(organism, nodes_local, node_velocity, previous_contact, phase, body_velocity)
        tensors = [torch.from_numpy(value[None]).to(self.device) for value in encoded]
        output = self.model(*tensors)
        appendages = len(organism.genome.appendages); muscles = len(organism.muscles)
        contact = (torch.sigmoid(output.contact_logits[0, :appendages]) >= .5).cpu().numpy()
        activation = output.muscle_activation[0, :muscles].float().cpu().numpy()
        drive = float(output.body_velocity[0].float().cpu()) * .55
        return activation.astype(np.float32), contact.astype(np.bool_), drive

