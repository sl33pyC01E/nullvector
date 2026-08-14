from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import torch
import torch.nn.functional as F

from ..map_topology_neural.codec import build_codec
from ..map_topology_neural_production.checkpoint import load_checkpoint
from ..map_topology_neural_production.contract import TopologyCodecCalibrationConfig
from ..map_topology_neural_production.dataset import TopologyProductionDataset, TopologyRef
from ..maps.model import THEMES
from .contract import (
    FROZEN_CODEC_CHECKPOINT_SHA256,
    FROZEN_CODEC_EMA_SHA256,
    FROZEN_CODEC_RELATIVE,
    FROZEN_CODEC_SOURCE_SHA256,
    PROJECT_ROOT,
    sha256_file,
)
from .masking import tensor_sha256


PATCH_SCALE: Final[int] = 4


@dataclass(frozen=True, slots=True)
class LatentBatchIdentity:
    sample_ids: tuple[str, ...]
    target_sha256: str
    valid_sha256: str
    point_sha256: str


class FrozenLatentDataset:
    def __init__(self, corpus_root: Path, checkpoint_path: Path | None = None) -> None:
        self.dataset = TopologyProductionDataset(Path(corpus_root))
        self.checkpoint_path = Path(checkpoint_path or PROJECT_ROOT / FROZEN_CODEC_RELATIVE).resolve()
        if sha256_file(self.checkpoint_path) != FROZEN_CODEC_CHECKPOINT_SHA256:
            raise ValueError("Masked-prior frozen codec checkpoint identity drifted.")
        payload = load_checkpoint(self.checkpoint_path)
        if payload["source_sha256"] != FROZEN_CODEC_SOURCE_SHA256:
            raise ValueError("Masked-prior frozen codec source identity drifted.")
        if payload["ema_state_sha256"] != FROZEN_CODEC_EMA_SHA256:
            raise ValueError("Masked-prior frozen codec EMA identity drifted.")
        codec_config = TopologyCodecCalibrationConfig.from_dict(payload["config"]).codec_config()
        self.codec = build_codec(codec_config, init_seed=payload["config"]["seed"])
        self.codec.load_state_dict(payload["ema_state"], strict=True)
        self.codec.eval()
        self.codec.requires_grad_(False)
        self.codec_payload = payload

    def smoke_refs(self) -> tuple[TopologyRef, ...]:
        selected: list[TopologyRef] = []
        for theme in THEMES:
            candidates = [
                ref for ref in self.dataset.refs_by_split["train"]
                if ref.theme == theme and ref.shape == (32, 32)
            ]
            if not candidates:
                raise ValueError(f"Masked-prior smoke has no 32x32 train reference for {theme}.")
            selected.append(min(candidates, key=lambda ref: ref.full_map_identity_sha256))
        return tuple(selected)

    def encode(self, refs: tuple[TopologyRef, ...]) -> tuple[dict[str, torch.Tensor], LatentBatchIdentity]:
        if not refs or len({ref.shape for ref in refs}) != 1:
            raise ValueError("Masked-prior latent batches must be nonempty and homogeneous.")
        batch = self.dataset.collate(refs, torch.device("cpu"))
        previous_threads = torch.get_num_threads()
        previous_deterministic = torch.are_deterministic_algorithms_enabled()
        previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            torch.set_num_threads(1)
            torch.use_deterministic_algorithms(True)
            with torch.inference_mode(), torch.backends.mkldnn.flags(enabled=False):
                encoded = self.codec.encode(batch, update_ema=False)
        finally:
            torch.use_deterministic_algorithms(previous_deterministic, warn_only=previous_warn_only)
            torch.set_num_threads(previous_threads)
        # Tensors created under inference_mode retain an inference-only flag;
        # clone outside that context before using targets in autograd losses.
        targets = encoded["indices"].detach().cpu().long().contiguous().clone()
        valid = F.max_pool2d(batch["valid_mask"].float(), PATCH_SCALE, PATCH_SCALE) > 0
        points = F.max_pool2d(batch["point_heatmaps"], PATCH_SCALE, PATCH_SCALE)
        result = {
            "targets": targets,
            "valid_mask": valid.bool().contiguous(),
            "point_conditions": points.float().contiguous(),
            "global_conditions": batch["global_conditions"].float().contiguous(),
            "theme_index": batch["theme_index"].long().contiguous(),
        }
        identity = LatentBatchIdentity(
            sample_ids=tuple(ref.full_map_identity_sha256 for ref in refs),
            target_sha256=tensor_sha256(targets),
            valid_sha256=tensor_sha256(valid),
            point_sha256=tensor_sha256(points),
        )
        return result, identity
