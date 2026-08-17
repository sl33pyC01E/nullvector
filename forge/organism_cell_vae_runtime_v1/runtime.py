from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from ..config import PROJECT_ROOT
from ..creature_stage_developmental.contract import APPENDAGE_KINDS, TISSUES
from ..organism_cell_vae_v1.contract import CELL_FEATURES, MAX_CELLS
from ..organism_cell_vae_v1.evaluation import validate as validate_release
from ..organism_cell_vae_v1.training import load_final


DEFAULT_RELEASE = PROJECT_ROOT / "outputs/organism_cell_vae_v1/production_v3_calibrated"


class ContinuousCellVAERuntime:
    """Validated continuous neural raster authority for posed cellular bodies."""

    def __init__(self, model, device: torch.device, alpha_threshold: float, release: Path):
        self.model = model
        self.device = device
        self.alpha_threshold = float(alpha_threshold)
        self.release = Path(release)

    @classmethod
    def from_release(cls, release: Path = DEFAULT_RELEASE, *, device: str = "cuda") -> "ContinuousCellVAERuntime":
        release = Path(release).resolve()
        checked = validate_release(release)
        if not checked.get("passed"):
            raise ValueError("continuous cell VAE release is not promoted")
        manifest = json.loads((release / "evaluation_manifest.json").read_text("utf-8"))
        if manifest.get("status") != "ready" or not all(value is True for value in manifest.get("gates", {}).values()):
            raise ValueError("continuous cell VAE quality gates drifted")
        target = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
        model, _, _ = load_final(release)
        return cls(model.to(target).eval(), target, manifest["evaluation"]["metrics"]["calibrated_alpha_threshold"], release)

    @staticmethod
    def organism_features(organism, cell_xy: np.ndarray, *, phase: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
        cell_xy = np.asarray(cell_xy, dtype=np.float32)
        count = int(organism.cell_count)
        if cell_xy.shape != (count, 2) or not 1 <= count <= MAX_CELLS or not np.isfinite(cell_xy).all():
            raise ValueError("continuous cell VAE posed-cell geometry drifted")
        family = int(np.argmax(np.asarray(organism.genome.family_mix, dtype=np.float32)))
        value = np.zeros((MAX_CELLS, CELL_FEATURES), np.float32)
        value[:count, :2] = cell_xy / 47 * 2 - 1
        value[np.arange(count), 2 + np.asarray(organism.tissue, dtype=np.int64)] = 1
        value[:count, 17 + family] = 1
        value[:count, 22:37] = np.asarray(organism.trait_fields, dtype=np.float32)
        value[:count, 37] = np.asarray(organism.appendage_index) < 0
        for cell, appendage_index in enumerate(np.asarray(organism.appendage_index)):
            if appendage_index >= 0:
                kind = organism.genome.appendages[int(appendage_index)].kind
                value[cell, 38 + APPENDAGE_KINDS.index(kind)] = 1
        value[:count, 46] = np.asarray(organism.side, dtype=np.float32)
        value[:count, 47] = math.sin(math.tau * phase)
        value[:count, 48] = math.cos(math.tau * phase)
        value[:count, 49] = np.asarray(organism.component_weights, dtype=np.float32).max(1)
        value[:count, 50] = np.asarray(organism.appendage_index) >= 0
        value[:count, 51] = 1
        mask = np.zeros(MAX_CELLS, np.bool_)
        mask[:count] = True
        return torch.from_numpy(value), torch.from_numpy(mask)

    @torch.inference_mode()
    def render_features(self, features, mask, *, threshold_alpha: bool = False) -> torch.Tensor:
        features = torch.as_tensor(features, dtype=torch.float32)
        mask = torch.as_tensor(mask, dtype=torch.bool)
        if features.ndim == 2:
            features, mask = features[None], mask[None]
        if features.ndim != 3 or features.shape[1:] != (MAX_CELLS, CELL_FEATURES) or mask.shape != features.shape[:2]:
            raise ValueError("continuous cell VAE runtime tensor geometry drifted")
        result = self.model(features.to(self.device), mask.to(self.device), stochastic=False).rgba.float().cpu()
        if threshold_alpha:
            result[:, 3:] = (result[:, 3:] >= self.alpha_threshold).to(result.dtype)
        return result

    def render_organism(self, organism, cell_xy: np.ndarray, *, phase: float = 0.0, threshold_alpha: bool = False) -> torch.Tensor:
        features, mask = self.organism_features(organism, cell_xy, phase=phase)
        return self.render_features(features, mask, threshold_alpha=threshold_alpha)[0]
