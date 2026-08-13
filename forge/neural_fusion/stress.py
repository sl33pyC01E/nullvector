from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from ..multifield_style_motion.hashing import canonical_json_bytes
from ..multifield_style_motion.io import require_disk_floor, write_exact
from ..neural_rig_repair.hashing import sha256_bytes
from ..neural_rig_repair.motion import compile_sample_motion_audit
from ..neural_rig_repair_style import load_repair_style_authority
from .genetics import fuse_specimen
from .hashing import stress_source_hash
from .pilot import DEFAULT_OUTPUT as DEFAULT_PILOT
from .rig import build_fusion_binding


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "neural_fusion_stress_v1"
FORMAT = "nullvector-neural-fusion-motion-stress-v1"
WORKER_FORMAT = "nullvector-neural-fusion-motion-stress-worker-v1"
MAX_WORKERS = 2
MAX_ATTEMPTS = 3
TIMEOUT_SECONDS = 900


def _load_pilot(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    payload = path.read_bytes()
    manifest = json.loads(payload)
    if canonical_json_bytes(manifest) != payload:
        raise ValueError("fusion pilot manifest must be canonical JSON")
    unsigned = dict(manifest)
    stored = unsigned.pop("bank_sha256", None)
    if stored != sha256_bytes(canonical_json_bytes(unsigned)):
        raise ValueError("fusion pilot manifest self-hash mismatch")
    if (
        manifest.get("status") != "ready"
        or manifest.get("counts", {}).get("specimen_count") != 10
        or any(value is not True for value in manifest.get("gates", {}).values())
    ):
        raise ValueError("fusion pilot is not ready for exhaustive stress")
    return manifest


def _rebuild_specimen(authority, record: dict[str, Any]):
    parent_a = authority.repair_source.samples[int(record["parent_a"]["ordinal"])]
    parent_b = authority.repair_source.samples[int(record["parent_b"]["ordinal"])]
    specimen = fuse_specimen(
        parent_a,
        parent_b,
        seed=int(record["seed"]),
        fusion_mode=str(record["fusion_mode"]),
        mutation_mode=str(record["mutation_mode"]),
        mutation_strength=int(record["mutation_strength"]),
        dominant_parent=str(record["dominant_parent"]),
    )
    if (
        specimen.genome.specimen_id != record["specimen_id"]
        or specimen.fields_sha256 != record["fields_sha256"]
        or specimen.provenance_sha256 != record["provenance_sha256"]
        or specimen.genome.lineage_sha256 != record["lineage_sha256"]
    ):
        raise ValueError("fusion stress specimen exact rebuild mismatch")
    return specimen


def compile_worker(pilot_manifest: Path, specimen_index: int, destination: Path) -> dict[str, Any]:
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError("fusion stress worker destination already exists")
    require_disk_floor(destination, planned_bytes=512 * 1024**2)
    pilot = _load_pilot(pilot_manifest)
    if not 0 <= specimen_index < len(pilot["specimens"]):
        raise ValueError("fusion stress specimen index is outside the pilot")
    authority = load_repair_style_authority()
    record = pilot["specimens"][specimen_index]
    specimen = _rebuild_specimen(authority, record)
    binding = build_fusion_binding(specimen)
    if binding.sha256 != record["binding_sha256"]:
        raise ValueError("fusion stress binding exact rebuild mismatch")
    motion = compile_sample_motion_audit(binding)
    result = {
        "format": WORKER_FORMAT,
        "status": "passed",
        "specimen_index": specimen_index,
        "specimen_id": specimen.genome.specimen_id,
        "fields_sha256": specimen.fields_sha256,
        "lineage_sha256": specimen.genome.lineage_sha256,
        "binding_sha256": binding.sha256,
        "motion_audit": motion,
        "gates": {
            "pilot_specimen_exactly_rebuilt": True,
            "binding_exactly_rebuilt": True,
            "all_104_motion_facing_clips_passed": True,
            "all_944_frames_passed": True,
            "loop_endpoints_exact": True,
            "source_tuples_preserved": True,
            "cpu_only": True,
        },
    }
    result["worker_sha256"] = sha256_bytes(canonical_json_bytes(result))
    destination.mkdir(parents=True)
    write_exact(destination / "motion_stress.json", canonical_json_bytes(result))
    return result


def _load_worker(path: Path, expected_index: int) -> dict[str, Any]:
    payload = path.read_bytes()
    record = json.loads(payload)
    if canonical_json_bytes(record) != payload:
        raise ValueError("fusion stress worker record is not canonical")
    unsigned = dict(record)
    stored = unsigned.pop("worker_sha256", None)
    if stored != sha256_bytes(canonical_json_bytes(unsigned)):
        raise ValueError("fusion stress worker self-hash mismatch")
    motion = record.get("motion_audit", {})
    if (
        record.get("format") != WORKER_FORMAT
        or record.get("status") != "passed"
        or record.get("specimen_index") != expected_index
        or motion.get("clip_count") != 104
        or motion.get("frame_count") != 944
        or len(motion.get("clips", [])) != 104
        or any(value is not True for value in record.get("gates", {}).values())
        or any(value is not True for value in motion.get("gates", {}).values())
    ):
        raise ValueError("fusion stress worker contract failed")
    return record


def _environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "-1",
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    return environment


def compile_stress(
    pilot_manifest: Path = DEFAULT_PILOT / "fusion_manifest.json",
    destination: Path = DEFAULT_OUTPUT,
    *,
    workers: int = MAX_WORKERS,
    max_attempts: int = MAX_ATTEMPTS,
) -> dict[str, Any]:
    destination = Path(destination).resolve()
    require_disk_floor(destination, planned_bytes=2 * 1024**3)
    if (destination / "fusion_stress_report.json").exists():
        raise FileExistsError("fusion stress output is already sealed")
    pilot = _load_pilot(pilot_manifest)
    destination.mkdir(parents=True, exist_ok=True)
    published = destination / "specimens"
    staging = destination / "staging"
    logs = destination / "logs"
    for path in (published, staging, logs):
        path.mkdir(parents=True, exist_ok=True)
    pending = []
    attempts = {index: 0 for index in range(10)}
    events: list[dict[str, Any]] = []
    for index in range(10):
        target = published / f"specimen_{index:02d}" / "motion_stress.json"
        if target.exists():
            _load_worker(target, index)
            events.append({"specimen_index": index, "attempt": 0, "status": "reused"})
        else:
            pending.append(index)
    active: dict[int, dict[str, Any]] = {}
    while pending or active:
        while pending and len(active) < min(max(1, workers), MAX_WORKERS):
            index = pending.pop(0)
            attempts[index] += 1
            attempt = attempts[index]
            if attempt > max_attempts:
                raise RuntimeError(f"fusion stress specimen {index} exhausted retries")
            stage = staging / f"specimen_{index:02d}_attempt_{attempt:02d}"
            log_path = logs / f"specimen_{index:02d}_attempt_{attempt:02d}.log"
            if stage.exists() or log_path.exists():
                raise FileExistsError("fusion stress attempt path already exists")
            handle = log_path.open("xb")
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "forge.neural_fusion.stress",
                    "worker",
                    "--pilot",
                    str(Path(pilot_manifest).resolve()),
                    "--specimen-index",
                    str(index),
                    "--output",
                    str(stage),
                ],
                cwd=PROJECT_ROOT,
                env=_environment(),
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            active[index] = {"process": process, "handle": handle, "stage": stage, "log": log_path, "attempt": attempt, "started": time.monotonic()}
        time.sleep(0.25)
        for index, state in list(active.items()):
            process = state["process"]
            elapsed = time.monotonic() - state["started"]
            return_code = process.poll()
            timed_out = return_code is None and elapsed > TIMEOUT_SECONDS
            if return_code is None and not timed_out:
                continue
            if timed_out:
                process.kill()
                return_code = process.wait(timeout=30)
            state["handle"].close()
            event = {
                "specimen_index": index,
                "attempt": state["attempt"],
                "return_code": int(return_code),
                "elapsed_seconds": round(elapsed, 3),
                "timed_out": timed_out,
                "access_violation": int(return_code) in {3221225477, -1073741819},
                "log": state["log"].relative_to(destination).as_posix(),
            }
            try:
                if return_code != 0 or timed_out:
                    raise RuntimeError("fusion stress worker failed")
                _load_worker(state["stage"] / "motion_stress.json", index)
                target = published / f"specimen_{index:02d}"
                if target.exists():
                    raise FileExistsError("fusion stress target appeared")
                os.replace(state["stage"], target)
                event["status"] = "published"
            except (OSError, RuntimeError, ValueError) as error:
                event["status"] = "rejected"
                event["error"] = str(error)[:1000]
                pending.append(index)
            events.append(event)
            del active[index]
    results = [
        _load_worker(published / f"specimen_{index:02d}" / "motion_stress.json", index)
        for index in range(10)
    ]
    report = {
        "format": FORMAT,
        "status": "passed",
        "source_sha256": stress_source_hash(),
        "pilot_bank_sha256": pilot["bank_sha256"],
        "counts": {
            "specimen_count": 10,
            "motion_count": 13,
            "facing_count": 8,
            "clip_count": 1040,
            "frame_count": 9440,
            "attempt_count": sum(attempts.values()),
            "retry_count": sum(max(0, value - 1) for value in attempts.values()),
            "access_violation_count": sum(bool(event.get("access_violation")) for event in events),
        },
        "specimens": [
            {
                "specimen_index": record["specimen_index"],
                "specimen_id": record["specimen_id"],
                "fields_sha256": record["fields_sha256"],
                "lineage_sha256": record["lineage_sha256"],
                "binding_sha256": record["binding_sha256"],
                "sample_motion_sha256": record["motion_audit"]["sample_motion_sha256"],
                "worker_sha256": record["worker_sha256"],
            }
            for record in results
        ],
        "events": events,
        "gates": {
            "all_ten_specimens_exactly_rebuilt": True,
            "all_ten_fresh_rigs_exactly_rebuilt": True,
            "all_1040_motion_facing_clips_passed": True,
            "all_9440_frames_passed": True,
            "all_loop_endpoints_exact": True,
            "all_source_tuples_preserved": True,
            "bounded_process_isolation": True,
            "disk_floor_preserved": True,
        },
    }
    report["stress_sha256"] = sha256_bytes(canonical_json_bytes(report))
    write_exact(destination / "fusion_stress_report.json", canonical_json_bytes(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Exhaustively stress neural fusion rigs")
    commands = parser.add_subparsers(dest="command", required=True)
    worker = commands.add_parser("worker")
    worker.add_argument("--pilot", type=Path, required=True)
    worker.add_argument("--specimen-index", type=int, required=True)
    worker.add_argument("--output", type=Path, required=True)
    compile_parser = commands.add_parser("compile")
    compile_parser.add_argument("--pilot", type=Path, default=DEFAULT_PILOT / "fusion_manifest.json")
    compile_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.command == "worker":
        result = compile_worker(args.pilot, args.specimen_index, args.output)
        print("NEURAL_FUSION_STRESS_WORKER_OK", result["specimen_index"], result["worker_sha256"])
    else:
        result = compile_stress(args.pilot, args.output)
        print("NEURAL_FUSION_STRESS_OK", result["counts"]["clip_count"], result["counts"]["frame_count"], result["stress_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
