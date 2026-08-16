from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch

from ..creature_stage_neural_grounded_controller.dataset import owner_metadata
from .contract import CHECKPOINT_FORMAT, GOALS, TARGET_TYPES, ModelConfig, source_sha256
from .model import NeuralGrasperController
from .dataset import feeder_anchor


@dataclass(frozen=True, slots=True)
class GraspCommand:
    appendage: int
    engage: bool
    reach: tuple[float, float]
    force: float
    target_type: str
    brace: float
    release: bool
    throw_impulse: tuple[float, float]


class NeuralGrasperRuntime:
    def __init__(self, model, device): self.model, self.device = model, device

    @classmethod
    def from_checkpoint(cls, path: Path, *, device="cuda"):
        target = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu"); payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("format") != CHECKPOINT_FORMAT or payload.get("status") != "evaluated" or payload.get("source_sha256") != source_sha256() or not payload.get("report", {}).get("gates", {}).get("all_passed"):
            raise ValueError("neural grasper provenance or quality drifted")
        model = NeuralGrasperController(ModelConfig(**payload["model_config"])); model.load_state_dict(payload["model_state"]); return cls(model.to(target).eval(), target)

    @torch.inference_mode()
    def plan(self, organism, *, target_type: str, goal: str, direction, distance: float, mass: float, cohesion: float, mobility: float, hostility: float = 0, throw: float = 0, attached: bool = False) -> GraspCommand:
        if target_type not in TARGET_TYPES or goal not in GOALS: raise ValueError("neural grasper target vocabulary drifted")
        owner, mask = owner_metadata(organism); target = np.zeros(18, np.float32); target[TARGET_TYPES.index(target_type)] = 1; target[4:6] = np.asarray(direction, np.float32); target[6:12] = (distance, mass, cohesion, mobility, hostility, throw); target[12 + GOALS.index(goal)] = 1; target[17] = float(attached)
        global_state = np.zeros(10, np.float32); global_state[int(np.argmax(np.asarray(organism.genome.family_mix, dtype=np.float32)))] = 1; traits = np.asarray(organism.genome.traits, np.float32); global_state[5:8] = (traits[3:8].mean(), traits[8:13].mean(), traits[13:18].mean()); global_state[8:10] = feeder_anchor(organism)
        result = self.model(torch.from_numpy(owner[None]).to(self.device), torch.from_numpy(mask[None]).to(self.device), torch.from_numpy(target[None]).to(self.device), torch.from_numpy(global_state[None]).to(self.device))
        reach = result.reach[0].float().cpu().numpy(); impulse = result.throw_impulse[0].float().cpu().numpy()
        return GraspCommand(int(result.appendage_logits[0].argmax()), bool(result.engage_logit[0] > 0), tuple(map(float, reach)), float(result.force[0]), TARGET_TYPES[int(result.type_logits[0].argmax())], float(result.brace[0]), bool(result.release_logit[0] > 0), tuple(map(float, impulse)))
