from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

from ..sprite_latent.codec import SemanticSpriteFSQ
from ..sprite_latent.training import canonical_state_hash
from ..sprite_latent_production.checkpoint import load_checkpoint
from ..sprite_latent_production.contract import ProductionConfig, sha256_file
from ..sprite_latent_production.supervisor import validate_production_manifest
from .contract import DEFAULT_PRODUCTION_MANIFEST


@dataclass(frozen=True, slots=True)
class ProductionCodecAuthority:
    model: SemanticSpriteFSQ
    manifest: dict[str, Any]
    manifest_path: Path
    manifest_file_sha256: str
    checkpoint_path: Path
    checkpoint_file_sha256: str
    ema_state_sha256: str


@lru_cache(maxsize=2)
def load_production_codec(manifest_path: str = str(DEFAULT_PRODUCTION_MANIFEST)) -> ProductionCodecAuthority:
    path = Path(manifest_path).resolve()
    manifest = validate_production_manifest(path)
    if manifest["status"] != "ready" or not manifest["gates"]["full_quality_accepted"]:
        raise ValueError("production fusion requires an accepted sprite latent authority")
    checkpoint_path = (path.parent / manifest["best"]["checkpoint"]).resolve()
    try:
        checkpoint_path.relative_to(path.parent)
    except ValueError as error:
        raise ValueError("production fusion checkpoint escapes its authority root") from error
    if checkpoint_path.is_symlink() or sha256_file(checkpoint_path) != manifest["best"]["checkpoint_sha256"]:
        raise ValueError("production fusion best checkpoint artifact mismatch")
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    model = SemanticSpriteFSQ(ProductionConfig.from_metadata(checkpoint["config"]).codec_config())
    model.load_state_dict(checkpoint["ema_state"], strict=True)
    if canonical_state_hash(model) != checkpoint["ema_state_sha256"]:
        raise ValueError("production fusion EMA semantic hash mismatch")
    model.eval()
    if any(parameter.device.type != "cpu" for parameter in model.parameters()) or torch.cuda.is_initialized():
        raise ValueError("production fusion codec loading must remain CPU-only")
    return ProductionCodecAuthority(
        model=model,
        manifest=manifest,
        manifest_path=path,
        manifest_file_sha256=sha256_file(path),
        checkpoint_path=checkpoint_path,
        checkpoint_file_sha256=sha256_file(checkpoint_path),
        ema_state_sha256=checkpoint["ema_state_sha256"],
    )
