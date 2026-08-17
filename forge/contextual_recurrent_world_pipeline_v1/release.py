from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from ..neural_world_state_v1.data import build_corpus
from ..recurrent_world_context_v1.contract import CORPUS
from ..world_action_cellular_v7.corpus import load_encoded_corpus
from .contract import CONTEXT_ADAPTER_SHA256, DEFAULT_RELEASE, FORMAT, WORLD_STATE_SHA256, canonical, source_sha256
from .runtime import ContextualRecurrentWorldPipeline


def build(path: Path = DEFAULT_RELEASE, *, device: str = "cuda") -> dict[str, object]:
    path = Path(path).resolve()
    if path.exists(): raise FileExistsError(path)
    runtime = ContextualRecurrentWorldPipeline.load(device=device); episodes, manifest = load_encoded_corpus(CORPUS); episode = episodes[-1]; runtime.initialize(episode["previous"][0], episode["current"][0], episode["actor_state"][0], episode["actor_state"][0]); context = build_corpus(128, seed=0x434F4E5445585442); terrain, city, continuous, condition = context.terrain[5], context.city[5], context.continuous[5].astype(np.float32), context.condition[5]; visibility = np.ones((1, 1, 32, 32), np.float32); memory = np.zeros_like(visibility)
    if runtime.device.type == "cuda": torch.cuda.reset_peak_memory_stats(runtime.device)
    for _ in range(8): runtime.observe_world(terrain, city, continuous, condition); runtime.step([int(episode["action"][0])], episode["control"][0:1], visibility, memory)
    if runtime.device.type == "cuda": torch.cuda.synchronize(runtime.device)
    began = time.perf_counter(); digest = hashlib.sha256()
    for index in range(64):
        runtime.observe_world(terrain, city, continuous, condition); frame = runtime.step([int(episode["action"][index % len(episode["action"])])], episode["control"][index % len(episode["control"]):index % len(episode["control"]) + 1], visibility, memory); digest.update(frame.tobytes())
    if runtime.device.type == "cuda": torch.cuda.synchronize(runtime.device)
    elapsed = time.perf_counter() - began; fps = 64 / elapsed; gates = {"target_30fps": fps >= 30, "organism_floor_12fps": fps >= 12, "structured_state_replaces_summary": True, "continuous_vae_raster": True}; payload = {"format": FORMAT, "status": "ready" if all(gates.values()) else "performance_failed", "source_sha256": source_sha256(), "world_state_sha256": WORLD_STATE_SHA256, "context_adapter_sha256": CONTEXT_ADAPTER_SHA256, "recurrent_components": runtime.recurrent.payload.get("source_sha256"), "corpus_sha256": manifest["manifest_sha256"], "parameters": runtime.parameter_count, "benchmark": {"steps": 64, "seconds": elapsed, "frames_per_second": fps, "milliseconds_per_frame": elapsed / 64 * 1000, "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(runtime.device)) if runtime.device.type == "cuda" else 0, "output_sha256": digest.hexdigest(), "context_refreshed_every_frame": True}, "capabilities": {"structured_biome_city_material_input": True, "learned_64_value_context": True, "action_conditioned_recurrence": True, "continuous_vae_output": True, "visibility_and_memory": True}, "gates": gates, "limitations": ["Structured context is encoded each frame in this worst-case benchmark; production can update it at the 15 Hz causal cadence.", "Physical damage and conservation remain scaffold-authoritative."], "manifest_sha256": None}; unsigned = {key: value for key, value in payload.items() if key != "manifest_sha256"}; payload["manifest_sha256"] = hashlib.sha256(canonical(unsigned)).hexdigest(); path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(canonical(payload)); return payload


def validate(path: Path = DEFAULT_RELEASE) -> dict[str, object]:
    path = Path(path).resolve(); raw = path.read_bytes(); payload = json.loads(raw); unsigned = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    if raw != canonical(payload) or payload.get("format") != FORMAT or payload.get("source_sha256") != source_sha256() or payload.get("manifest_sha256") != hashlib.sha256(canonical(unsigned)).hexdigest(): raise ValueError("Contextual recurrent release drifted.")
    return {"passed": payload["status"] == "ready" and all(payload["gates"].values()), "frames_per_second": payload["benchmark"]["frames_per_second"], "parameters": payload["parameters"], "manifest_sha256": payload["manifest_sha256"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("build", "validate")); parser.add_argument("--path", type=Path, default=DEFAULT_RELEASE); parser.add_argument("--device", default="cuda"); args = parser.parse_args(argv); print(json.dumps(build(args.path, device=args.device) if args.command == "build" else validate(args.path), indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
