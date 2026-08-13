from __future__ import annotations

import argparse
from collections import Counter, deque
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping

from ..morphology.motion import DEFAULT_FRAME_COUNTS, FACING_NAMES, MOTION_NAMES
from ..neural_rig_bridge.model import DRIVER_NAMES
from .binding import bind_repair_plan
from .constants import (
    DEFAULT_GENERATION_MANIFEST,
    DEFAULT_STYLE_MANIFEST,
    DISK_FLOOR_BYTES,
    EXPECTED_BRIDGE_SOURCE_SHA256,
    EXPECTED_CLIP_COUNT,
    EXPECTED_FRAME_COUNT,
    EXPECTED_SAMPLE_COUNT,
    MAX_STRESS_SHARD_BYTES,
    MAX_STRESS_REPORT_BYTES,
    STRESS_FORMAT,
    STRESS_MAX_ATTEMPTS,
    STRESS_SHARD_COUNT,
    STRESS_SHARD_FORMAT,
    STRESS_TIMEOUT_SECONDS,
    STRESS_WORKERS,
)
from .hashing import (
    artifact_record,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    source_hash,
    write_canonical_json,
)
from .motion import compile_sample_motion_audit
from .planner import load_repair_plan
from .schema import load_strict_json, resolve_artifact_record
from .source import load_repair_source


def _unsigned_hash(payload: Mapping[str, Any], hash_key: str) -> str:
    unsigned = dict(payload)
    stored = unsigned.pop(hash_key)
    expected = sha256_bytes(canonical_json_bytes(unsigned))
    if stored != expected:
        raise ValueError(f"{hash_key} self-hash mismatch")
    return expected


def _validate_clip(clip: Mapping[str, Any], sample_id: str) -> None:
    expected_keys = {
        "format",
        "sample_id",
        "binding_sha256",
        "motion",
        "facing",
        "fps",
        "loop",
        "frame_count",
        "motion_strength",
        "z_order",
        "z_order_policy",
        "fit_source_to_destination",
        "unfitted_matrix_sequence_sha256",
        "fitted_matrix_sequence_sha256",
        "frame_sequence_sha256",
        "frames",
        "metrics",
        "gates",
        "clip_sha256",
    }
    if not isinstance(clip, Mapping) or set(clip) != expected_keys:
        raise ValueError("stress clip registry keys are not exact")
    if (
        not isinstance(clip["frames"], list)
        or not isinstance(clip["metrics"], Mapping)
        or not isinstance(clip["gates"], Mapping)
        or not isinstance(clip["z_order"], list)
    ):
        raise ValueError("stress clip nested registry types are malformed")
    if (
        clip["format"] != "nullvector-neural-rig-repair-clip-audit-v1"
        or clip["sample_id"] != sample_id
        or clip["motion"] not in MOTION_NAMES
        or clip["facing"] not in FACING_NAMES
        or clip["frame_count"] != DEFAULT_FRAME_COUNTS[clip["motion"]]
        or len(clip["frames"]) != clip["frame_count"]
        or len(clip["z_order"]) != len(DRIVER_NAMES)
        or set(clip["z_order"]) != set(DRIVER_NAMES)
        or clip["z_order_policy"]
        not in {"frozen-default", "clip-local-driver-retention"}
        or any(value is not True for value in clip["metrics"].values() if isinstance(value, bool))
        or any(value is not True for value in clip["gates"].values())
    ):
        raise ValueError("stress clip contract failed")
    if set(clip["metrics"]) != {
        "minimum_foreground_pixels",
        "maximum_foreground_pixels",
        "minimum_driver_pixels",
        "maximum_driver_pixels",
        "maximum_anchor_support_distance",
        "all_frames_margin_clear",
        "all_frames_source_tuples_only",
        "all_frames_retain_every_driver",
        "all_transformed_anchors_have_local_driver_support",
    }:
        raise ValueError("stress clip metric registry keys are not exact")
    for index, frame in enumerate(clip["frames"]):
        if not isinstance(frame, Mapping) or set(frame) != {
            "index",
            "frame_sha256",
            "posed_fields_sha256",
            "driver_index_sha256",
            "foreground_pixels",
            "driver_pixels",
        }:
            raise ValueError("stress frame registry keys are not exact")
        if frame["index"] != index or len(frame["driver_pixels"]) != 8:
            raise ValueError("stress frame registry is out of order")
    if clip["frame_sequence_sha256"] != sha256_bytes(
        canonical_json_bytes(clip["frames"])
    ):
        raise ValueError("stress clip frame sequence hash mismatch")
    _unsigned_hash(clip, "clip_sha256")


def _validate_sample_motion(payload: Mapping[str, Any], sample_id: str) -> None:
    if not isinstance(payload, Mapping) or set(payload) != {
        "format",
        "sample_id",
        "binding_sha256",
        "clip_count",
        "frame_count",
        "motion_count",
        "facing_count",
        "strength_histogram",
        "clips",
        "gates",
        "sample_motion_sha256",
    }:
        raise ValueError("stress sample-motion keys are not exact")
    if (
        payload["format"]
        != "nullvector-neural-rig-repair-sample-motion-audit-v1"
        or payload["sample_id"] != sample_id
        or payload["clip_count"] != 104
        or payload["frame_count"] != 944
        or payload["motion_count"] != 13
        or payload["facing_count"] != 8
        or len(payload["clips"]) != 104
        or any(value is not True for value in payload["gates"].values())
    ):
        raise ValueError("stress sample-motion contract failed")
    expected_order = [
        (motion, facing) for motion in MOTION_NAMES for facing in FACING_NAMES
    ]
    for clip, expected in zip(payload["clips"], expected_order, strict=True):
        _validate_clip(clip, sample_id)
        if (clip["motion"], clip["facing"]) != expected:
            raise ValueError("stress clips are not in canonical motion/facing order")
    if sum(int(clip["frame_count"]) for clip in payload["clips"]) != 944:
        raise ValueError("stress sample frame count mismatch")
    _unsigned_hash(payload, "sample_motion_sha256")


def load_stress_shard(path: Path) -> dict[str, Any]:
    shard = load_strict_json(
        path,
        maximum_bytes=MAX_STRESS_SHARD_BYTES,
        label="neural rig repair stress shard",
    )
    if set(shard) != {
        "format",
        "status",
        "shard_index",
        "shard_count",
        "source",
        "sample_count",
        "clip_count",
        "frame_count",
        "samples",
        "gates",
        "shard_sha256",
    }:
        raise ValueError("stress shard keys are not exact")
    if (
        shard["format"] != STRESS_SHARD_FORMAT
        or shard["status"] != "passed"
        or type(shard["shard_index"]) is not int
        or shard["shard_count"] != STRESS_SHARD_COUNT
        or not 0 <= shard["shard_index"] < shard["shard_count"]
        or shard["sample_count"] != len(shard["samples"])
        or shard["clip_count"] != shard["sample_count"] * 104
        or shard["frame_count"] != shard["sample_count"] * 944
        or any(value is not True for value in shard["gates"].values())
    ):
        raise ValueError("stress shard contract failed")
    expected_ordinals = list(
        range(shard["shard_index"], EXPECTED_SAMPLE_COUNT, shard["shard_count"])
    )
    if [sample["ordinal"] for sample in shard["samples"]] != expected_ordinals:
        raise ValueError("stress shard sample assignment is not canonical")
    for sample in shard["samples"]:
        if not isinstance(sample, Mapping) or set(sample) != {
            "sample_id",
            "ordinal",
            "family",
            "raw_fields_sha256",
            "plan_sha256",
            "binding_sha256",
            "baseline_v1_status",
            "baseline_v1_categories",
            "motion_audit",
        }:
            raise ValueError("stress sample registry keys are not exact")
        _validate_sample_motion(sample["motion_audit"], sample["sample_id"])
        if sample["motion_audit"]["binding_sha256"] != sample["binding_sha256"]:
            raise ValueError("stress sample binding hash linkage mismatch")
    _unsigned_hash(shard, "shard_sha256")
    return shard


def load_stress_report(path: Path, *, verify_shards: bool = True) -> dict[str, Any]:
    report = load_strict_json(
        path,
        maximum_bytes=MAX_STRESS_REPORT_BYTES,
        label="neural rig repair stress report",
    )
    if set(report) != {
        "format",
        "status",
        "source",
        "shard_count",
        "sample_count",
        "clip_count",
        "frame_count",
        "strength_histogram",
        "shards",
        "sample_results",
        "gates",
        "stress_sha256",
    }:
        raise ValueError("stress report keys are not exact")
    if (
        report["format"] != STRESS_FORMAT
        or report["status"] != "passed"
        or report["shard_count"] != STRESS_SHARD_COUNT
        or report["sample_count"] != EXPECTED_SAMPLE_COUNT
        or report["clip_count"] != EXPECTED_CLIP_COUNT
        or report["frame_count"] != EXPECTED_FRAME_COUNT
        or len(report["shards"]) != STRESS_SHARD_COUNT
        or len(report["sample_results"]) != EXPECTED_SAMPLE_COUNT
        or sum(int(value) for value in report["strength_histogram"].values())
        != EXPECTED_CLIP_COUNT
        or any(value is not True for value in report["gates"].values())
    ):
        raise ValueError("stress report contract failed")
    if [result["ordinal"] for result in report["sample_results"]] != list(
        range(EXPECTED_SAMPLE_COUNT)
    ):
        raise ValueError("stress report sample registry is not canonical")
    _unsigned_hash(report, "stress_sha256")
    if verify_shards:
        seen: list[int] = []
        for index, record in enumerate(report["shards"]):
            shard_path = resolve_artifact_record(
                Path(path).resolve().parent,
                record,
                label=f"stress shard {index}",
                maximum_bytes=MAX_STRESS_SHARD_BYTES,
            )
            shard = load_stress_shard(shard_path)
            if shard["shard_index"] != index:
                raise ValueError("stress report shard ordering mismatch")
            if shard["source"] != report["source"]:
                raise ValueError("stress report shard source linkage mismatch")
            seen.extend(sample["ordinal"] for sample in shard["samples"])
        if sorted(seen) != list(range(EXPECTED_SAMPLE_COUNT)):
            raise ValueError("stress report shards do not cover all 80 samples once")
    return report


def compile_stress_shard(
    generation_manifest: Path,
    style_manifest: Path,
    plan_directory: Path,
    shard_index: int,
    destination: Path,
) -> dict[str, Any]:
    if not 0 <= shard_index < STRESS_SHARD_COUNT:
        raise ValueError("stress shard index is out of bounds")
    source = load_repair_source(generation_manifest, style_manifest)
    samples = []
    for ordinal in range(shard_index, EXPECTED_SAMPLE_COUNT, STRESS_SHARD_COUNT):
        if shutil.disk_usage(destination.parent).free < DISK_FLOOR_BYTES:
            raise RuntimeError("100 GiB free-space floor reached during repair stress")
        sample = source.samples[ordinal]
        plan = load_repair_plan(plan_directory / f"{sample.sample_id}.repair.json")
        binding = bind_repair_plan(source, sample, plan, verify_exact_plan=True)
        motion_audit = compile_sample_motion_audit(binding)
        samples.append(
            {
                "sample_id": sample.sample_id,
                "ordinal": sample.ordinal,
                "family": sample.family,
                "raw_fields_sha256": sample.raw_fields_sha256,
                "plan_sha256": plan["hashes"]["plan_sha256"],
                "binding_sha256": binding.sha256,
                "baseline_v1_status": plan["baseline_v1"]["status"],
                "baseline_v1_categories": list(plan["baseline_v1"]["categories"]),
                "motion_audit": motion_audit,
            }
        )
    base: dict[str, Any] = {
        "format": STRESS_SHARD_FORMAT,
        "status": "passed",
        "shard_index": shard_index,
        "shard_count": STRESS_SHARD_COUNT,
        "source": {
            "generation_manifest_sha256": source.generation_manifest_sha256,
            "style_manifest_sha256": source.style_manifest_sha256,
            "frozen_bridge_source_sha256": EXPECTED_BRIDGE_SOURCE_SHA256,
            "repair_source_sha256": source_hash(),
        },
        "sample_count": len(samples),
        "clip_count": len(samples) * 104,
        "frame_count": len(samples) * 944,
        "samples": samples,
        "gates": {
            "canonical_modulo_shard_assignment": True,
            "cpu_only_environment": os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
            and os.environ.get("NEURAL_RIG_REPAIR_CPU_ONLY") == "1",
            "all_plans_exactly_recompiled": True,
            "all_bindings_valid": True,
            "all_motion_audits_passed": True,
        },
    }
    if any(value is not True for value in base["gates"].values()):
        raise ValueError("stress worker environment or result gate failed")
    base["shard_sha256"] = sha256_bytes(canonical_json_bytes(base))
    write_canonical_json(destination, base)
    return base


def _exit_telemetry(return_code: int | None) -> tuple[int | None, str]:
    if return_code is None:
        return None, "timeout"
    unsigned = return_code & 0xFFFFFFFF
    if return_code == 0:
        return unsigned, "success"
    if unsigned == 0xC0000005:
        return unsigned, "access_violation"
    if unsigned == 0xC0000409:
        return unsigned, "stack_buffer_overrun"
    if unsigned == 0xC00000FD:
        return unsigned, "stack_overflow"
    if unsigned in {0xC000001D, 0xC0000094}:
        return unsigned, "hardware_exception"
    if return_code < 0:
        return unsigned, "signal"
    return unsigned, "nonzero_exit"


def run_process_sharded_stress(
    generation_manifest: Path,
    style_manifest: Path,
    plan_directory: Path,
    destination: Path,
    *,
    workers: int = STRESS_WORKERS,
    timeout_seconds: int = STRESS_TIMEOUT_SECONDS,
    max_attempts: int = STRESS_MAX_ATTEMPTS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("stress workers must be an integer in [1, 4]")
    if type(timeout_seconds) is not int or not 30 <= timeout_seconds <= 3600:
        raise ValueError("stress timeout must be an integer in [30, 3600]")
    if type(max_attempts) is not int or not 1 <= max_attempts <= 5:
        raise ValueError("stress max attempts must be an integer in [1, 5]")
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    shard_directory = destination / "shards"
    shard_directory.mkdir(parents=True, exist_ok=True)
    pending = deque(range(STRESS_SHARD_COUNT))
    attempts: Counter[int] = Counter()
    active: dict[int, dict[str, Any]] = {}
    telemetry_records: list[dict[str, Any]] = []
    failed: dict[int, str] = {}

    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "-1",
            "NEURAL_RIG_REPAIR_CPU_ONLY": "1",
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    expected_shard_source = {
        "generation_manifest_sha256": sha256_file(Path(generation_manifest).resolve()),
        "style_manifest_sha256": sha256_file(Path(style_manifest).resolve()),
        "frozen_bridge_source_sha256": EXPECTED_BRIDGE_SOURCE_SHA256,
        "repair_source_sha256": source_hash(),
    }
    while pending or active:
        while pending and len(active) < workers:
            shard_index = pending.popleft()
            shard_path = shard_directory / f"stress_shard_{shard_index:02d}.json"
            if shard_path.exists():
                try:
                    existing = load_stress_shard(shard_path)
                    if existing["shard_index"] != shard_index:
                        raise ValueError("existing shard index mismatch")
                    if existing["source"] != expected_shard_source:
                        raise ValueError("existing shard source binding is stale")
                    telemetry_records.append(
                        {
                            "shard_index": shard_index,
                            "attempt": 0,
                            "result": "reused_exact_valid_shard",
                            "return_code_signed": 0,
                            "return_code_unsigned": 0,
                            "exit_category": "success",
                            "duration_seconds": 0.0,
                            "stdout_tail": "",
                            "stderr_tail": "",
                        }
                    )
                    continue
                except ValueError as error:
                    raise ValueError(
                        f"Refusing invalid pre-existing stress shard {shard_path}: {error}"
                    ) from error
            attempts[shard_index] += 1
            command = [
                sys.executable,
                "-m",
                "forge.neural_rig_repair.stress",
                "worker",
                "--generation",
                str(Path(generation_manifest).resolve()),
                "--style",
                str(Path(style_manifest).resolve()),
                "--plans",
                str(Path(plan_directory).resolve()),
                "--shard-index",
                str(shard_index),
                "--destination",
                str(shard_path),
            ]
            process = subprocess.Popen(
                command,
                cwd=str(Path(__file__).resolve().parents[2]),
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            active[shard_index] = {
                "process": process,
                "started": time.monotonic(),
                "attempt": attempts[shard_index],
                "path": shard_path,
            }

        if not active:
            continue
        time.sleep(0.1)
        for shard_index in list(active):
            state = active[shard_index]
            process: subprocess.Popen[str] = state["process"]
            duration = time.monotonic() - state["started"]
            timed_out = duration > timeout_seconds and process.poll() is None
            if timed_out:
                process.kill()
            return_code = process.poll()
            if return_code is None and not timed_out:
                continue
            stdout, stderr = process.communicate()
            if timed_out:
                return_code = None
            unsigned, category = _exit_telemetry(return_code)
            result = "passed" if return_code == 0 else "failed"
            record = {
                "shard_index": shard_index,
                "attempt": state["attempt"],
                "result": result,
                "return_code_signed": return_code,
                "return_code_unsigned": unsigned,
                "exit_category": category,
                "duration_seconds": round(duration, 6),
                "stdout_tail": stdout[-4000:],
                "stderr_tail": stderr[-8000:],
            }
            telemetry_records.append(record)
            del active[shard_index]
            if return_code == 0:
                try:
                    loaded = load_stress_shard(state["path"])
                    if loaded["shard_index"] != shard_index:
                        raise ValueError("worker returned a mismatched shard")
                except Exception as error:
                    record["result"] = "invalid_output"
                    record["stderr_tail"] += f"\nSupervisor validation: {error}"
                    return_code = 1
            if return_code != 0:
                if attempts[shard_index] < max_attempts:
                    pending.append(shard_index)
                else:
                    failed[shard_index] = record["stderr_tail"][-2000:]

    telemetry = {
        "format": "nullvector-neural-rig-repair-stress-telemetry-v1",
        "worker_limit": workers,
        "timeout_seconds": timeout_seconds,
        "max_attempts": max_attempts,
        "cpu_only_environment": True,
        "records": telemetry_records,
        "failed_shards": [
            {"shard_index": index, "stderr_tail": error}
            for index, error in sorted(failed.items())
        ],
    }
    write_canonical_json(
        destination / "stress_telemetry.json", telemetry, replace=True
    )
    if failed:
        raise RuntimeError(f"repair motion stress exhausted retries for shards {sorted(failed)}")

    shards = [
        load_stress_shard(shard_directory / f"stress_shard_{index:02d}.json")
        for index in range(STRESS_SHARD_COUNT)
    ]
    sample_records = [sample for shard in shards for sample in shard["samples"]]
    sample_records.sort(key=lambda sample: sample["ordinal"])
    if [sample["ordinal"] for sample in sample_records] != list(range(EXPECTED_SAMPLE_COUNT)):
        raise ValueError("stress aggregate does not contain each source sample exactly once")
    strength_histogram: Counter[str] = Counter()
    for sample in sample_records:
        strength_histogram.update(sample["motion_audit"]["strength_histogram"])
    report_base: dict[str, Any] = {
        "format": STRESS_FORMAT,
        "status": "passed",
        "source": dict(shards[0]["source"]),
        "shard_count": STRESS_SHARD_COUNT,
        "sample_count": len(sample_records),
        "clip_count": sum(int(shard["clip_count"]) for shard in shards),
        "frame_count": sum(int(shard["frame_count"]) for shard in shards),
        "strength_histogram": dict(sorted(strength_histogram.items())),
        "shards": [
            artifact_record(
                shard_directory / f"stress_shard_{index:02d}.json", destination
            )
            for index in range(STRESS_SHARD_COUNT)
        ],
        "sample_results": [
            {
                "sample_id": sample["sample_id"],
                "ordinal": sample["ordinal"],
                "binding_sha256": sample["binding_sha256"],
                "sample_motion_sha256": sample["motion_audit"]["sample_motion_sha256"],
            }
            for sample in sample_records
        ],
        "gates": {
            "all_16_shards_passed": True,
            "all_80_samples_once": True,
            "all_8320_clips_passed": True,
            "all_75520_frames_passed": True,
            "no_cuda_visible_devices": True,
            "crash_telemetry_captured": True,
        },
    }
    if (
        report_base["sample_count"] != EXPECTED_SAMPLE_COUNT
        or report_base["clip_count"] != EXPECTED_CLIP_COUNT
        or report_base["frame_count"] != EXPECTED_FRAME_COUNT
        or sum(strength_histogram.values()) != EXPECTED_CLIP_COUNT
        or any(value is not True for value in report_base["gates"].values())
    ):
        raise ValueError("stress aggregate count or gate failure")
    report_base["stress_sha256"] = sha256_bytes(canonical_json_bytes(report_base))
    write_canonical_json(destination / "stress_report.json", report_base)
    return report_base, telemetry


def _worker_main(args: argparse.Namespace) -> int:
    try:
        compile_stress_shard(
            args.generation,
            args.style,
            args.plans,
            args.shard_index,
            args.destination,
        )
        return 0
    except Exception:
        failure = {
            "format": "nullvector-neural-rig-repair-stress-failure-v1",
            "shard_index": args.shard_index,
            "pid": os.getpid(),
            "traceback": traceback.format_exc()[-16000:],
        }
        failure_path = Path(args.destination).with_suffix(".failure.json")
        try:
            write_canonical_json(failure_path, failure, replace=True)
        except Exception:
            pass
        traceback.print_exc()
        return 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m forge.neural_rig_repair.stress")
    subparsers = parser.add_subparsers(dest="command", required=True)
    worker = subparsers.add_parser("worker")
    worker.add_argument("--generation", type=Path, default=DEFAULT_GENERATION_MANIFEST)
    worker.add_argument("--style", type=Path, default=DEFAULT_STYLE_MANIFEST)
    worker.add_argument("--plans", type=Path, required=True)
    worker.add_argument("--shard-index", type=int, required=True)
    worker.add_argument("--destination", type=Path, required=True)
    return parser


if __name__ == "__main__":
    arguments = _parser().parse_args()
    if arguments.command == "worker":
        raise SystemExit(_worker_main(arguments))
    raise SystemExit(2)
