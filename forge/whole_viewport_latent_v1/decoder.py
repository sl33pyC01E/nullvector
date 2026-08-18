from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from ..recurrent_world_pipeline_v1.runtime import RecurrentWorldPipeline
from ..whole_viewport_raster_vae_v1.contract import canonical as raster_canonical
from ..whole_viewport_raster_vae_v1.contract import source_sha256 as raster_source_sha256
from ..world_frame_vae.contract import ModelConfig as DecoderConfig
from ..world_frame_vae.model import WorldFrameVAE


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_decoder(device: torch.device, release: Path | None = None):
    if release is None:
        decoder = RecurrentWorldPipeline.load(str(device)).decoder.eval()
        return decoder, {"kind": "foundation", "release": None}

    release = Path(release).resolve()
    raw = (release / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    if raw != raster_canonical(manifest):
        raise ValueError("raster VAE manifest is not canonical")
    if manifest.get("status") != "accepted" or not manifest.get("gates", {}).get("all_passed"):
        raise ValueError("raster VAE release is not accepted")
    if manifest.get("source_sha256") != raster_source_sha256():
        raise ValueError("raster VAE source provenance drifted")
    artifact = release / manifest["artifact"]["path"]
    if artifact.stat().st_size != manifest["artifact"]["bytes"] or _file_sha256(artifact) != manifest["artifact"]["sha256"]:
        raise ValueError("raster VAE artifact drifted")
    payload = torch.load(artifact, map_location="cpu", weights_only=False)
    if payload.get("status") != "accepted" or payload.get("source_sha256") != manifest["source_sha256"]:
        raise ValueError("raster VAE payload drifted")
    decoder = WorldFrameVAE(DecoderConfig(**payload["model_config"]))
    decoder.load_state_dict(payload["state"])
    decoder.to(device).eval()
    provenance = {
        "kind": "adapted_whole_viewport",
        "release": str(release),
        "manifest_sha256": manifest["manifest_sha256"],
        "artifact_sha256": manifest["artifact"]["sha256"],
        "selection": manifest["selection"],
    }
    return decoder, provenance
