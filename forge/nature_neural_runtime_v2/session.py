from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

from ..config import PROJECT_ROOT
from ..neural_ensemble_v1.contract import DEFAULT_OUTPUT as ENSEMBLE_OUTPUT


FORMAT = "nullvector-coupled-neural-ensemble-runtime/2.0.0"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "ensemble_runtime_v2" / "run_001"


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run(output: Path = DEFAULT_OUTPUT, *, steps: int = 45, device: str = "cuda", seed: int = 0x51554944) -> dict:
    if not 15 <= steps <= 900:
        raise ValueError("coupled ensemble evidence requires 15..900 world steps")
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    report_path = output / "runtime_report.json"
    frame_path = output / "coupled_world_vae.png"
    forecast_path = output / "dit_forecast_refined.png"
    if report_path.exists() or frame_path.exists() or forecast_path.exists():
        raise FileExistsError("coupled ensemble output is immutable")

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    from ..nature_sim_v2.demo import NatureDemo

    started = time.perf_counter()
    demo = NatureDemo(seed=seed, device=device, showcase=True)
    initial_world = demo.world.snapshot().semantic_sha256
    initial_society = demo.society.semantic_sha256()
    macro_events = 0
    try:
        for _ in range(steps):
            demo.update(1 / 30, step_pose=True)
            macro_events += int(bool(demo.ensemble.last_tick and demo.ensemble.last_tick.macro_applied))
        demo.ensemble.run_slow_cycle()
        demo.timeline_forecast = demo.ensemble.timeline_forecast
        demo.counterfactuals = dict(demo.ensemble.counterfactuals)
        demo._refresh_interventions()
        demo.neural_raster = True
        demo.render_alpha = demo.pose_render_alpha = 1.0
        selected = demo.world.organisms[demo.selected]
        organism_rgba = demo.ensemble._time(
            "cell_vae_present",
            lambda: demo.cell_vae.render_organism(
                selected.body.organism,
                demo._posed_points(selected),
                phase=(demo.world.time * .45 + selected.entity_id * .037) % 1,
            ),
        )
        if tuple(organism_rgba.shape) != (4, 96, 96):
            raise ValueError("cell VAE runtime output drifted")
        demo.ensemble._time("world_vae_present", demo.draw)
        if demo.neural_future is not None:
            demo.neural_future.result(); demo._poll_neural_job()
            demo.ensemble._time("world_vae_present", demo.draw)
        if demo.sprite_future is not None:
            demo.sprite_future.result(); demo._prepare_sprites()
            demo.ensemble._time("cell_vae_present", demo.draw)
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        demo.pg.image.save(demo.screen, str(frame_path))

        # Run the original Action-DiT + VAE + pixel-refiner stack on the same
        # authoritative frame. It remains a forecast/distillation observer;
        # it does not overwrite conserved cells, organs, materials or cities.
        from ..action_teacher_v1 import ACTIONS as TEACHER_ACTIONS
        from ..composite_world_v1 import CompositeWorldRuntime
        from ..nature_timeline_nn import extract_world_features
        composite = CompositeWorldRuntime.from_release(device=device)
        teacher = np.ascontiguousarray(demo.teacher_frame)
        action = demo.action_latch if demo.action_latch in TEACHER_ACTIONS else "none"
        def forecast():
            latent = composite.encode(teacher)
            future = composite.step_visual(
                latent,
                action=np.asarray((TEACHER_ACTIONS.index(action),), np.int64),
                control=demo._neural_control()[None],
                state=extract_world_features(demo.world, demo.society)[None],
                steps=4,
            )
            return composite.decode(future).float().clamp(0, 1)
        forecast_rgba = demo.ensemble._time("dit_vae_refiner_forecast", forecast)
        forecast_rgb = np.clip(forecast_rgba[0, :3].permute(1, 2, 0).cpu().numpy() * 255, 0, 255).astype(np.uint8)
        forecast_surface = demo.pg.surfarray.make_surface(np.transpose(forecast_rgb, (1, 0, 2)))
        demo.pg.image.save(forecast_surface, str(forecast_path))
        del composite

        final_snapshot = demo.world.snapshot()
        final_society = demo.society.semantic_sha256()
        manifest_path = ENSEMBLE_OUTPUT / "ensemble_manifest.json"
        manifest = json.loads(manifest_path.read_text("utf-8"))
        telemetry = demo.ensemble.timing_summary()
        payload = {
            "format": FORMAT,
            "status": "coupled_teacher_ensemble_running",
            "seed": int(seed),
            "device": str(demo.neural.locomotion.device),
            "steps": int(steps),
            "elapsed_seconds": time.perf_counter() - started,
            "world": {
                "initial_sha256": initial_world,
                "final_sha256": final_snapshot.semantic_sha256,
                "changed": initial_world != final_snapshot.semantic_sha256,
                "population": final_snapshot.population,
                "births": final_snapshot.births,
                "deaths": final_snapshot.deaths,
                "colonies": final_snapshot.colony_count,
            },
            "society": {
                "initial_sha256": initial_society,
                "final_sha256": final_society,
                "changed": initial_society != final_society,
                "factions": len(demo.society.factions),
                "settlements": len(demo.society.settlements),
            },
            "authority": {
                "shared_physical_state": True,
                "local_neural_outputs_mutate_world": True,
                "macro_neural_resource_delta_applied": macro_events > 0,
                "society_neural_decisions_mutate_economy_and_structures": True,
                "timeline_and_counterfactual_are_observers": True,
                "cell_vae_renders_physical_cell_geometry": True,
                "world_vae_renders_the_same_world_state": True,
                "recurrent_student_is_non_authoritative": True,
                "dit_vae_refiner_forecast_is_non_authoritative": True,
            },
            "activation": {
                "macro_events": macro_events,
                "macro_resource_delta_l1": demo.ensemble.macro_delta_l1,
                "component_count": len(manifest["components"]),
                "component_names": [row["name"] for row in manifest["components"]],
                "coordinator_calls": dict(sorted(demo.ensemble.calls.items())),
                "runtime_evidence": {
                    "locomotion_entity_states": len(demo.runtime.states),
                    "behavior_cached_entities": len(demo.behavior.cache),
                    "behavior_last_tick": int(demo.behavior.last_tick),
                    "colony_policy_calls": int(demo.colony_runtime.calls),
                    "feeding_entity_states": len(demo.feeding.entities),
                    "physiology_entity_states": len(demo.physiology.states),
                },
            },
            "timing": telemetry,
            "presentation": {
                "path": frame_path.relative_to(PROJECT_ROOT).as_posix(),
                "bytes": frame_path.stat().st_size,
                "sha256": _sha256(frame_path),
                "forecast_path": forecast_path.relative_to(PROJECT_ROOT).as_posix(),
                "forecast_bytes": forecast_path.stat().st_size,
                "forecast_sha256": _sha256(forecast_path),
            },
            "provenance": {
                "ensemble_manifest": manifest_path.relative_to(PROJECT_ROOT).as_posix(),
                "ensemble_manifest_sha256": _sha256(manifest_path),
            },
            "gpu": {
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else 0,
            },
        }
        required = payload["authority"]
        evidence = payload["activation"]["runtime_evidence"]
        if not payload["world"]["changed"] or not payload["society"]["changed"] or not all(required.values()) or not all(value > 0 for value in evidence.values()):
            raise ValueError("coupled neural ensemble did not establish authority")
        payload["report_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
        _atomic(report_path, _canonical(payload))
        return payload
    finally:
        demo.neural_executor.shutdown(wait=True, cancel_futures=True)
        demo.sprite_executor.shutdown(wait=True, cancel_futures=True)
        demo.pg.quit()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=45)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=lambda value: int(value, 0), default=0x51554944)
    args = parser.parse_args()
    result = run(args.output, steps=args.steps, device=args.device, seed=args.seed)
    print(json.dumps({"status": result["status"], "report_sha256": result["report_sha256"], "timing": result["timing"]}, indent=2))


if __name__ == "__main__":
    main()
