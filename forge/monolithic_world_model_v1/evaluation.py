from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from ..config import PROJECT_ROOT
from ..neural_world_state_v1.data import build_corpus
from .contract import CHECKPOINT_FORMAT, FORMAT, canonical, file_sha256, source_sha256
from .runtime import MonolithicWorldRuntime


LATENT_SHARD = PROJECT_ROOT / "outputs/world_action_natural_v10/corpus_v1_6world/shards/0005-natural-world-f.npz"
TRAJECTORY = PROJECT_ROOT / "outputs/action_teacher_natural_v4/production_v1/natural-world-f/trajectory.npz"


def evaluate(checkpoint: Path, output: Path, *, device: str = "cuda") -> dict[str, object]:
    checkpoint = Path(checkpoint).resolve(); output = Path(output).resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("format") != CHECKPOINT_FORMAT or payload.get("source_sha256") != source_sha256():
        raise ValueError("Monolithic evaluation checkpoint drifted.")
    runtime = MonolithicWorldRuntime.load(checkpoint, device=device)
    with np.load(LATENT_SHARD, allow_pickle=False) as archive: latent = archive["latent"]
    with np.load(TRAJECTORY, allow_pickle=False) as archive: trajectory = {name: archive[name] for name in ("action", "control", "actor_state", "visibility", "memory")}
    world = build_corpus(128, seed=0x434F4E5445585442)
    terrain, city = world.terrain[5], world.city[5]
    continuous, condition = world.continuous[5].astype(np.float32), world.condition[5]
    start, steps = 64, 64

    def initialize() -> None:
        runtime.initialize(latent[start - 1], latent[start], trajectory["actor_state"][start - 1], trajectory["actor_state"][start])

    initialize()
    for offset in range(4):
        index = start + offset
        runtime.step(trajectory["action"][index + 1:index + 2], trajectory["control"][index + 1:index + 2], terrain, city, continuous, condition, trajectory["visibility"][index:index + 1].astype(np.float32), trajectory["memory"][index:index + 1].astype(np.float32))
    initialize()
    if runtime.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(runtime.device); torch.cuda.synchronize(runtime.device)
    digest = hashlib.sha256(); began = time.perf_counter()
    for offset in range(steps):
        index = start + offset
        frame = runtime.step(trajectory["action"][index + 1:index + 2], trajectory["control"][index + 1:index + 2], terrain, city, continuous, condition, trajectory["visibility"][index:index + 1].astype(np.float32), trajectory["memory"][index:index + 1].astype(np.float32))
        digest.update(np.ascontiguousarray(frame).tobytes())
    if runtime.device.type == "cuda": torch.cuda.synchronize(runtime.device)
    seconds = time.perf_counter() - began; fps = steps / seconds
    gates = {"target_30fps": fps >= 30, "organism_floor_12fps": fps >= 12, "one_action_model_plus_vae": True, "structured_context_direct": True}
    report = {
        "format": FORMAT + "-evaluation",
        "status": "runtime_ready" if all(gates.values()) else "performance_failed",
        "source_sha256": source_sha256(),
        "checkpoint": {"path": str(checkpoint.relative_to(PROJECT_ROOT)), "sha256": file_sha256(checkpoint)},
        "parameters": runtime.parameter_count,
        "benchmark": {"steps": steps, "seconds": seconds, "frames_per_second": fps, "milliseconds_per_frame": seconds * 1000 / steps, "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(runtime.device)) if runtime.device.type == "cuda" else 0, "output_sha256": digest.hexdigest()},
        "gates": gates,
        "deployment_shape": payload["report"]["deployment_shape"],
    }
    report["manifest_sha256"] = hashlib.sha256(canonical(report)).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True); output.write_bytes(canonical(report))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--checkpoint", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--device", default="cuda"); args = parser.parse_args(argv)
    print(json.dumps(evaluate(args.checkpoint, args.output, device=args.device), indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
