from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .contract import canonical, source_sha256
from .recorder import validate_trajectory


FAMILIES = tuple(range(5))
SCENARIOS = ("journey", "migration", "feeding", "predation", "injury", "settlement_pan")


def _specs(frames: int, base_seed: int):
    for family in FAMILIES:
        for scenario_index, scenario in enumerate(SCENARIOS):
            seed = base_seed + family * 1009 + scenario_index * 97
            session = f"f{family}_{scenario}_{frames}_s{seed:08x}"
            yield session, family, scenario, seed


def _valid(root: Path, session: str, family: int, scenario: str, frames: int) -> bool:
    path = root / session
    if not path.is_dir():
        return False
    try:
        manifest = validate_trajectory(path)
        report = json.loads((path / "curriculum_report.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False
    return manifest["frames"] == frames and report["actor_family"] == family and report["scenario"] == scenario


def _run_one(root: Path, spec, frames: int, device: str, attempts: int):
    session, family, scenario, seed = spec
    if _valid(root, session, family, scenario, frames):
        return {"session": session, "status": "reused", "attempts": 0}
    telemetry = []
    for attempt in range(1, attempts + 1):
        started = time.monotonic()
        command = [sys.executable, "-m", "forge.action_teacher_viewport_v5", "--root", str(root), "--session", session, "--frames", str(frames), "--seed", str(seed), "--device", device, "--family", str(family), "--scenario", scenario]
        environment = dict(os.environ, PYTHONHASHSEED="0", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", NUMEXPR_NUM_THREADS="2")
        process = subprocess.run(command, cwd=Path(__file__).resolve().parents[2], env=environment, capture_output=True, text=True)
        record = {"attempt": attempt, "exit_code": process.returncode, "seconds": round(time.monotonic() - started, 3), "stderr_tail": process.stderr[-2000:]}
        telemetry.append(record)
        if process.returncode == 0 and _valid(root, session, family, scenario, frames):
            return {"session": session, "family": family, "scenario": scenario, "seed": seed, "status": "built", "attempts": attempt, "telemetry": telemetry}
    return {"session": session, "family": family, "scenario": scenario, "seed": seed, "status": "failed", "attempts": attempts, "telemetry": telemetry}


def build(root: Path, *, frames=384, base_seed=0x56504543, workers=4, attempts=2, device="cuda"):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if not 1 <= workers <= 6 or not 1 <= attempts <= 3:
        raise ValueError("whole-viewport corpus resource policy drifted")
    specs = tuple(_specs(frames, base_seed))
    results = []
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="viewport-corpus") as executor:
        futures = {executor.submit(_run_one, root, spec, frames, device, attempts): spec for spec in specs}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps({"finished": len(results), "total": len(specs), **result}), flush=True)
    results.sort(key=lambda item: item["session"])
    if any(item["status"] == "failed" for item in results):
        raise RuntimeError("whole-viewport corpus has failed strata")
    manifests = [validate_trajectory(root / session) for session, *_ in specs]
    report = {"format": "nullvector-whole-viewport-corpus/5.0.0", "source_sha256": source_sha256(), "frames_per_session": frames, "sessions": len(specs), "frames": frames * len(specs), "families": list(FAMILIES), "scenarios": list(SCENARIOS), "workers": workers, "device": device, "results": results, "trajectory_manifests": [item["manifest_sha256"] for item in manifests]}
    report["corpus_sha256"] = hashlib.sha256(canonical(report)).hexdigest()
    (root / "corpus_report.json").write_bytes(canonical(report))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=384)
    parser.add_argument("--base-seed", type=lambda value: int(value, 0), default=0x56504543)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(json.dumps(build(args.root, frames=args.frames, base_seed=args.base_seed, workers=args.workers, attempts=args.attempts, device=args.device), indent=2))


if __name__ == "__main__":
    main()
