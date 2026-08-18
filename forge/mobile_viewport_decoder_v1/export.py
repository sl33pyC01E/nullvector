from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from .contract import ModelConfig, canonical, config_dict, source_sha256
from .model import MobileViewportDecoder


EXPORT_FORMAT = "nullvector-mobile-viewport-decoder-onnx/1.0.0"


def export_onnx(release: Path, destination: Path) -> dict:
    release, destination = Path(release), Path(destination)
    manifest = json.loads((release / "manifest.json").read_bytes())
    if manifest.get("status") != "accepted" or not manifest.get("gates", {}).get("all_passed"):
        raise ValueError("mobile viewport decoder release is not accepted")
    if manifest.get("source_sha256") != source_sha256():
        raise ValueError("mobile viewport decoder source provenance drifted")
    artifact = release / manifest["artifact"]["path"]
    if hashlib.sha256(artifact.read_bytes()).hexdigest() != manifest["artifact"]["sha256"]:
        raise ValueError("mobile viewport decoder artifact hash drifted")
    payload = torch.load(artifact, map_location="cpu", weights_only=False)
    config = ModelConfig(**manifest["model_config"])
    model = MobileViewportDecoder(config).eval()
    model.load_state_dict(payload["state"], strict=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (torch.zeros(1, config.latent_channels, 32, 32),),
        destination,
        input_names=["latent"],
        output_names=["rgb"],
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
    )
    result = {
        "format": EXPORT_FORMAT,
        "source_sha256": source_sha256(),
        "release_manifest_sha256": manifest["manifest_sha256"],
        "model_config": config_dict(config),
        "artifact": {
            "path": destination.name,
            "bytes": destination.stat().st_size,
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        },
    }
    result["manifest_sha256"] = hashlib.sha256(canonical(result)).hexdigest()
    destination.with_suffix(".manifest.json").write_bytes(canonical(result))
    return result
