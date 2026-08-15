from __future__ import annotations

import numpy as np
import torch

from ..creature_stage_neural_grounded_cyclic.curriculum import CurriculumGroundedTeacher
from ..creature_stage_neural_grounded_cyclic.dataset import RuntimeHonestGroundedTeacher
from .contract import MAX_APPENDAGES


def _owners(arrays: dict[str, np.ndarray], identities: torch.Tensor, device: torch.device) -> torch.Tensor:
    values = np.stack([arrays["appendage_owner"][int(identity)] for identity in identities.cpu().tolist()]).astype(np.int64)
    if values.min() < -1 or values.max() >= MAX_APPENDAGES:
        raise ValueError("component owner vocabulary drifted")
    return torch.from_numpy(values).to(device)


class ComponentCurriculumTeacher(CurriculumGroundedTeacher):
    def batch(self, step: int, batch_size: int, device: torch.device, *, split: str = "train", frame_offset: int = 0):
        result = super().batch(step, batch_size, device, split=split, frame_offset=frame_offset)
        result["owner"] = _owners(self.arrays, result["identity"], device)
        return result


class ComponentSentinelTeacher(RuntimeHonestGroundedTeacher):
    def owner(self, identity: int, device: torch.device) -> torch.Tensor:
        return torch.from_numpy(self.arrays["appendage_owner"][identity].astype(np.int64, copy=True))[None].to(device)
