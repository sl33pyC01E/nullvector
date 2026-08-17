from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch

from ..maps.io import file_sha256
from .contract import CHECKPOINT_FORMAT, GrowthModelConfig, canonical_json_bytes, source_sha256
from .model import build_model


RELEASE_FORMAT = "nullvector-neural-city-growth-ema-v1/1.0.0"


def release(checkpoint: Path, destination: Path, *, device: str = "cuda") -> dict[str, object]:
    checkpoint = Path(checkpoint).resolve(); destination = Path(destination).resolve()
    if destination.exists(): raise FileExistsError(destination)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("format") != CHECKPOINT_FORMAT or payload.get("source_sha256") != source_sha256():
        raise ValueError("Growth checkpoint provenance drifted.")
    config = GrowthModelConfig(**payload["model_config"])
    model = build_model(config); model.load_state_dict(payload["ema_state"], strict=True)
    target = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu"); model = model.to(target).eval()
    current = torch.zeros((1, 24, 24), dtype=torch.long, device=target)
    condition = torch.zeros((1, model.condition[0].in_features), device=target)
    with torch.inference_mode(), torch.autocast(target.type, dtype=torch.bfloat16, enabled=target.type == "cuda"):
        for _ in range(20): model(current, condition)
        if target.type == "cuda": torch.cuda.synchronize(target)
        started = time.perf_counter()
        for _ in range(200): model(current, condition)
        if target.type == "cuda": torch.cuda.synchronize(target)
    elapsed = time.perf_counter() - started
    artifact = {"format": RELEASE_FORMAT, "source_sha256": payload["source_sha256"], "parent_checkpoint_sha256": file_sha256(checkpoint), "model_config": asdict(config), "ema_state": {name: value.detach().cpu() for name, value in payload["ema_state"].items()}}
    destination.parent.mkdir(parents=True, exist_ok=True); torch.save(artifact, destination)
    report = {"format": RELEASE_FORMAT, "artifact": destination.name, "artifact_sha256": file_sha256(destination), "artifact_bytes": destination.stat().st_size, "parameters": sum(value.numel() for value in model.parameters()), "single_tick_milliseconds": elapsed / 200 * 1000, "single_tick_hz": 200 / elapsed, "device": str(target), "runtime_vram_allocated_bytes": int(torch.cuda.max_memory_allocated(target)) if target.type == "cuda" else 0}
    destination.with_suffix(".json").write_bytes(canonical_json_bytes(report)); return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--checkpoint", type=Path, required=True); parser.add_argument("--destination", type=Path, required=True); parser.add_argument("--device", default="cuda"); args = parser.parse_args(argv); print(json.dumps(release(args.checkpoint, args.destination, device=args.device), indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
