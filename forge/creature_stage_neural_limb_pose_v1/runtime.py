from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..creature_stage_neural_motion.training import _state_sha256
from .contract import CHECKPOINT_FORMAT, ModelConfig, source_sha256
from .dataset import encode_case
from .model import NeuralLimbPose


class NeuralLimbPoseDriver:
    """Causal neural muscle-pose driver with exact physical boundary pins."""

    def __init__(self, model: NeuralLimbPose, device: torch.device) -> None:
        self.model = model.eval(); self.device = device

    @classmethod
    def from_checkpoint(cls, path: Path, *, device: str = "cuda") -> "NeuralLimbPoseDriver":
        target = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        if payload.get("format") != CHECKPOINT_FORMAT or payload.get("source_sha256") != source_sha256():
            raise ValueError("neural limb pose checkpoint provenance drifted")
        state = payload.get("model_state")
        if not isinstance(state, dict) or _state_sha256(state) != payload.get("model_state_sha256"):
            raise ValueError("neural limb pose state drifted")
        if not payload.get("report", {}).get("gates", {}).get("all_passed"):
            raise ValueError("neural limb pose checkpoint failed quality")
        model = NeuralLimbPose(ModelConfig(**payload["model_config"])); model.load_state_dict(state, strict=True)
        return cls(model.to(target), target)

    @torch.inference_mode()
    def predict_pose(
        self, organism, appendage, positions, velocities, root, target, lengths, *,
        response: float, actuation: float, load: float, bend_sign: float,
    ) -> np.ndarray:
        nodes, context, mask, reach = encode_case(
            organism, appendage, positions, velocities, root, target, lengths,
            response=response, actuation=actuation, load=load, bend_sign=bend_sign,
        )
        output = self.model(
            torch.from_numpy(nodes[None]).to(self.device),
            torch.from_numpy(context[None]).to(self.device),
            torch.from_numpy(mask[None]).to(self.device),
        )
        count = len(lengths) + 1
        pose = output.pose[0, :count].float().cpu().numpy() * reach + np.asarray(root, np.float32)
        pose[0] = root; pose[-1] = target
        return pose.astype(np.float32)
