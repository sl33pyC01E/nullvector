from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Sequence

from ..config import OUTPUT_DIR, PROJECT_ROOT
from ..safety import require_disk_floor, write_json_atomic
from .generator import generate_map, splitmix64
from .io import array_digest, write_map_pack
from .model import MAP_SCHEMA_VERSION, MapConfig, THEMES
from .validate import validate_map, validate_pack


def _semantic_identity_digest(indices: Sequence[int], hashes: Sequence[str]) -> str:
    if len(indices) != len(hashes):
        raise ValueError("semantic identity indices and hashes must align")
    digest = hashlib.sha256()
    for index, semantic_hash in sorted(zip(indices, hashes, strict=True)):
        digest.update(int(index).to_bytes(8, "little", signed=False))
        digest.update(bytes.fromhex(semantic_hash))
    return digest.hexdigest()


def _theme_list(values: Sequence[str]) -> list[str]:
    if not values or "all" in values:
        return list(THEMES)
    result: list[str] = []
    for value in values:
        if value not in THEMES:
            raise ValueError(f"Unknown theme {value!r}; expected one of {THEMES} or all.")
        if value not in result:
            result.append(value)
    return result


def fuzz_maps(
    count: int,
    *,
    base_seed: int = 0x4D4150464F524745,
    width: int = 64,
    height: int = 64,
    start_index: int = 0,
    include_hashes: bool = False,
) -> dict[str, object]:
    if count < 1:
        raise ValueError("Fuzz count must be positive.")
    if start_index < 0:
        raise ValueError("Fuzz start_index must be non-negative.")
    require_disk_floor(PROJECT_ROOT, planned_bytes=0)
    started = time.perf_counter()
    failures: list[dict[str, object]] = []
    per_theme = {theme: 0 for theme in THEMES}
    repair_total = 0
    separations: list[int] = []
    walkable_ratios: list[float] = []
    hashes: set[str] = set()
    semantic_hashes: list[str] = []
    processed_indices: list[int] = []

    for local_index in range(count):
        index = start_index + local_index
        theme = THEMES[index % len(THEMES)]
        seed = splitmix64(base_seed ^ index ^ ((index % len(THEMES) + 1) << 48))
        # Three nearby dimensions catch parity and rectangular-layout mistakes.
        width_delta = (index % 3) * 4
        height_delta = ((index // 3) % 3) * 4
        cfg = MapConfig(width=width + width_delta, height=height + height_delta)
        try:
            data = generate_map(seed, theme, cfg)
            report = validate_map(data)
            if not report["passed"]:
                failures.append(
                    {"index": index, "theme": theme, "seed": seed, "failures": report["failures"]}
                )
                continue
            per_theme[theme] += 1
            repair_total += int(data.repair_count)
            separations.append(int(report["metrics"]["start_exit_path_length"]))
            walkable_ratios.append(float(report["metrics"]["walkable_ratio"]))
            # Keep canonical, constant-size all-field identities rather than
            # retaining every full semantic array pair during long fuzz runs. A
            # 5k-map Windows process otherwise held tens of megabytes of Python
            # bytes objects and intermittently ended in an allocator-adjacent
            # access violation inside a later NumPy conversion.
            semantic_hash = array_digest(data.arrays())
            hashes.add(semantic_hash)
            semantic_hashes.append(semantic_hash)
            processed_indices.append(index)
        except Exception as error:
            failures.append({"index": index, "theme": theme, "seed": seed, "error": repr(error)})

    elapsed = time.perf_counter() - started
    passed_count = count - len(failures)
    report: dict[str, object] = {
        "passed": not failures,
        "requested": count,
        "start_index": start_index,
        "passed_count": passed_count,
        "failure_count": len(failures),
        "per_theme": per_theme,
        "unique_semantic_maps": len(hashes),
        "repair_total": repair_total,
        "minimum_start_exit_path_length": min(separations, default=-1),
        "maximum_start_exit_path_length": max(separations, default=-1),
        "minimum_walkable_ratio": round(min(walkable_ratios, default=0.0), 6),
        "maximum_walkable_ratio": round(max(walkable_ratios, default=0.0), 6),
        "elapsed_seconds": round(elapsed, 3),
        "maps_per_second": round(passed_count / max(elapsed, 1e-9), 3),
        "semantic_identity_sha256": _semantic_identity_digest(
            processed_indices, semantic_hashes
        ),
        "failures": failures[:50],
    }
    if include_hashes:
        report["semantic_hashes"] = semantic_hashes
        report["processed_indices"] = processed_indices
    return report


def _stderr_tail(value: str, limit: int = 4000) -> str:
    return value[-limit:] if len(value) > limit else value


def _validate_worker_report(
    report: object,
    *,
    expected_start: int,
    expected_count: int,
) -> str | None:
    if not isinstance(report, dict):
        return "worker payload is not an object"
    required = {
        "passed",
        "requested",
        "start_index",
        "passed_count",
        "failure_count",
        "per_theme",
        "semantic_hashes",
        "semantic_identity_sha256",
        "processed_indices",
        "repair_total",
        "minimum_start_exit_path_length",
        "maximum_start_exit_path_length",
        "minimum_walkable_ratio",
        "maximum_walkable_ratio",
        "failures",
    }
    missing = sorted(required - set(report))
    if missing:
        return f"worker payload is missing keys: {missing}"
    if not isinstance(report["passed"], bool):
        return "passed is not boolean"
    if report["requested"] != expected_count or report["start_index"] != expected_start:
        return "worker range disagrees with the requested chunk"
    if isinstance(report["passed_count"], bool) or not isinstance(report["passed_count"], int):
        return "passed_count is not an integer"
    if not 0 <= report["passed_count"] <= expected_count:
        return "passed_count is outside the chunk range"
    if report["failure_count"] != expected_count - report["passed_count"]:
        return "failure_count does not complement passed_count"
    if not isinstance(report["failures"], list):
        return "failures is not a list"
    indices = report["processed_indices"]
    if not isinstance(indices, list) or any(
        isinstance(index, bool) or not isinstance(index, int) for index in indices
    ):
        return "processed_indices is not an integer list"
    if len(indices) != report["passed_count"] or len(set(indices)) != len(indices):
        return "processed_indices count or uniqueness is invalid"
    expected_indices = set(range(expected_start, expected_start + expected_count))
    if not set(indices) <= expected_indices:
        return "processed_indices escapes the requested chunk"
    if report["passed"] and set(indices) != expected_indices:
        return "passing worker omitted one or more chunk indices"
    semantic_hashes = report["semantic_hashes"]
    if not isinstance(semantic_hashes, list) or len(semantic_hashes) != len(indices):
        return "semantic_hashes does not align with processed_indices"
    if any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in semantic_hashes
    ):
        return "semantic_hashes contains a non-SHA256 value"
    if report["semantic_identity_sha256"] != _semantic_identity_digest(
        indices, semantic_hashes
    ):
        return "semantic_identity_sha256 disagrees with indexed hashes"
    per_theme = report["per_theme"]
    if (
        not isinstance(per_theme, dict)
        or set(per_theme) != set(THEMES)
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in per_theme.values())
        or sum(per_theme.values()) != report["passed_count"]
    ):
        return "per_theme counts are malformed or incomplete"
    if report["passed"] and (
        report["passed_count"] != expected_count
        or report["failure_count"] != 0
        or report["failures"]
    ):
        return "passing worker has inconsistent pass/failure fields"
    for name in (
        "repair_total",
        "minimum_start_exit_path_length",
        "maximum_start_exit_path_length",
    ):
        if isinstance(report[name], bool) or not isinstance(report[name], int):
            return f"{name} is not an integer"
    for name in ("minimum_walkable_ratio", "maximum_walkable_ratio"):
        if isinstance(report[name], bool) or not isinstance(report[name], (int, float)):
            return f"{name} is not numeric"
    return None


def fuzz_maps_isolated(
    count: int,
    *,
    base_seed: int = 0x4D4150464F524745,
    width: int = 64,
    height: int = 64,
    chunk_size: int = 250,
    worker_retries: int = 2,
    worker_timeout_seconds: int = 300,
    _inject_worker_exit_start_index: int | None = None,
) -> dict[str, object]:
    """Run long fuzz gates in bounded fresh processes and aggregate exactly.

    Windows hosts in this project have exhibited rare, nondeterministic access
    violations after several thousand mixed NumPy operations.  Each worker is
    therefore disposable: a native crash is recorded and retried in a fresh
    interpreter, while repeated failure remains a hard gate failure.  Global
    indices preserve the original seed, theme, and dimension sequence exactly.
    """
    if count < 1:
        raise ValueError("Fuzz count must be positive.")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")
    if worker_retries < 0:
        raise ValueError("worker_retries must be non-negative.")
    if worker_timeout_seconds < 1:
        raise ValueError("worker_timeout_seconds must be positive.")
    require_disk_floor(PROJECT_ROOT, planned_bytes=0)
    started = time.perf_counter()
    per_theme = {theme: 0 for theme in THEMES}
    repair_total = 0
    minimum_separation: int | None = None
    maximum_separation: int | None = None
    minimum_walkable: float | None = None
    maximum_walkable: float | None = None
    hashes: set[str] = set()
    indexed_hashes: dict[int, str] = {}
    failures: list[dict[str, object]] = []
    worker_attempts: list[dict[str, object]] = []
    passed_count = 0
    processed_indices: set[int] = set()

    for start_index in range(0, count, chunk_size):
        requested = min(chunk_size, count - start_index)
        chunk_report: dict[str, object] | None = None
        for attempt in range(1, worker_retries + 2):
            command = [
                sys.executable,
                "-X",
                "faulthandler",
                "-m",
                "forge.maps",
                "fuzz-worker",
                "--count",
                str(requested),
                "--start-index",
                str(start_index),
                "--seed",
                hex(base_seed),
                "--width",
                str(width),
                "--height",
                str(height),
            ]
            if _inject_worker_exit_start_index == start_index and attempt == 1:
                command.extend(("--inject-exit-code", "86"))
            attempt_started = time.perf_counter()
            try:
                completed = subprocess.run(
                    command,
                    cwd=PROJECT_ROOT,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=worker_timeout_seconds,
                    check=False,
                )
                elapsed = time.perf_counter() - attempt_started
                attempt_record: dict[str, object] = {
                    "start_index": start_index,
                    "requested": requested,
                    "attempt": attempt,
                    "exit_code": completed.returncode,
                    "elapsed_seconds": round(elapsed, 3),
                }
                if completed.stderr:
                    attempt_record["stderr_tail"] = _stderr_tail(completed.stderr)
                worker_attempts.append(attempt_record)
                if completed.returncode != 0:
                    continue
                try:
                    candidate = json.loads(completed.stdout)
                except json.JSONDecodeError as error:
                    attempt_record["parse_error"] = str(error)
                    attempt_record["stdout_tail"] = _stderr_tail(completed.stdout)
                    continue
                protocol_error = _validate_worker_report(
                    candidate,
                    expected_start=start_index,
                    expected_count=requested,
                )
                if protocol_error is not None:
                    attempt_record["protocol_error"] = protocol_error
                    continue
                if not candidate.get("passed"):
                    attempt_record["contract_failures"] = candidate.get("failures", [])
                    # Invariant failures are deterministic evidence, not a
                    # transient native crash: do not obscure them with retries.
                    chunk_report = candidate
                    break
                chunk_report = candidate
                break
            except subprocess.TimeoutExpired as error:
                worker_attempts.append(
                    {
                        "start_index": start_index,
                        "requested": requested,
                        "attempt": attempt,
                        "timed_out": True,
                        "timeout_seconds": worker_timeout_seconds,
                        "stdout_tail": _stderr_tail(
                            error.stdout.decode("utf-8", errors="replace")
                            if isinstance(error.stdout, bytes)
                            else (error.stdout or "")
                        ),
                        "stderr_tail": _stderr_tail(
                            error.stderr.decode("utf-8", errors="replace")
                            if isinstance(error.stderr, bytes)
                            else (error.stderr or "")
                        ),
                    }
                )

        if chunk_report is None:
            failures.append(
                {
                    "kind": "worker_failed_after_retries",
                    "start_index": start_index,
                    "requested": requested,
                    "attempts": worker_retries + 1,
                }
            )
            continue
        if not chunk_report.get("passed"):
            failures.extend(chunk_report.get("failures", []))
            continue

        chunk_indices = set(int(index) for index in chunk_report["processed_indices"])
        overlap = processed_indices & chunk_indices
        if overlap:
            failures.append(
                {
                    "kind": "worker_index_overlap",
                    "start_index": start_index,
                    "overlap": sorted(overlap),
                }
            )
            continue
        processed_indices.update(chunk_indices)

        passed_count += int(chunk_report["passed_count"])
        repair_total += int(chunk_report["repair_total"])
        for theme in THEMES:
            per_theme[theme] += int(chunk_report["per_theme"][theme])
        chunk_min_separation = int(chunk_report["minimum_start_exit_path_length"])
        chunk_max_separation = int(chunk_report["maximum_start_exit_path_length"])
        chunk_min_walkable = float(chunk_report["minimum_walkable_ratio"])
        chunk_max_walkable = float(chunk_report["maximum_walkable_ratio"])
        minimum_separation = (
            chunk_min_separation
            if minimum_separation is None
            else min(minimum_separation, chunk_min_separation)
        )
        maximum_separation = (
            chunk_max_separation
            if maximum_separation is None
            else max(maximum_separation, chunk_max_separation)
        )
        minimum_walkable = (
            chunk_min_walkable
            if minimum_walkable is None
            else min(minimum_walkable, chunk_min_walkable)
        )
        maximum_walkable = (
            chunk_max_walkable
            if maximum_walkable is None
            else max(maximum_walkable, chunk_max_walkable)
        )
        for index, semantic_hash in zip(
            chunk_report["processed_indices"],
            chunk_report["semantic_hashes"],
            strict=True,
        ):
            indexed_hashes[int(index)] = str(semantic_hash)
            hashes.add(str(semantic_hash))

    elapsed = time.perf_counter() - started
    expected_indices = set(range(count))
    missing_indices = sorted(expected_indices - processed_indices)
    unexpected_indices = sorted(processed_indices - expected_indices)
    if missing_indices or unexpected_indices:
        failures.append(
            {
                "kind": "aggregate_index_coverage",
                "missing_indices": missing_indices[:100],
                "unexpected_indices": unexpected_indices[:100],
                "missing_count": len(missing_indices),
                "unexpected_count": len(unexpected_indices),
            }
        )
    return {
        "passed": not failures and passed_count == count,
        "requested": count,
        "passed_count": passed_count,
        "failure_count": count - passed_count,
        "per_theme": per_theme,
        "unique_semantic_maps": len(hashes),
        "repair_total": repair_total,
        "minimum_start_exit_path_length": minimum_separation if minimum_separation is not None else -1,
        "maximum_start_exit_path_length": maximum_separation if maximum_separation is not None else -1,
        "minimum_walkable_ratio": round(minimum_walkable or 0.0, 6),
        "maximum_walkable_ratio": round(maximum_walkable or 0.0, 6),
        "elapsed_seconds": round(elapsed, 3),
        "maps_per_second": round(passed_count / max(elapsed, 1e-9), 3),
        "semantic_identity_sha256": _semantic_identity_digest(
            list(indexed_hashes), list(indexed_hashes.values())
        ),
        "isolation": {
            "enabled": True,
            "chunk_size": chunk_size,
            "worker_retries": worker_retries,
            "worker_timeout_seconds": worker_timeout_seconds,
            "worker_count": (count + chunk_size - 1) // chunk_size,
            "attempt_count": len(worker_attempts),
            "retry_count": sum(int(record.get("attempt", 1)) > 1 for record in worker_attempts),
            "processed_index_count": len(processed_indices),
            "attempts": worker_attempts,
        },
        "failures": failures[:50],
    }


def _generate(args: argparse.Namespace) -> int:
    themes = _theme_list(args.themes)
    cfg = MapConfig(
        width=args.width,
        height=args.height,
        objective_count=args.objectives,
        spawn_count=args.spawns,
        min_start_exit_distance=args.min_separation,
        spawn_clearance_start=args.spawn_start_clearance,
        spawn_clearance_objective=args.spawn_objective_clearance,
        spawn_clearance_hazard=args.spawn_hazard_clearance,
    )
    output = Path(args.output).resolve()
    require_disk_floor(output, planned_bytes=len(themes) * args.count * 4 * 1024 * 1024)
    generated: list[str] = []
    started = time.perf_counter()
    for theme_index, theme in enumerate(themes):
        for index in range(args.count):
            seed = splitmix64(args.seed ^ ((theme_index + 1) << 48) ^ index)
            data = generate_map(seed, theme, cfg)
            pack = write_map_pack(
                data,
                output,
                preview_scale=args.preview_scale,
                skip_existing=args.skip_existing,
            )
            generated.append(str(pack))
    payload = {
        "passed": True,
        "output": str(output),
        "pack_count": len(generated),
        "themes": themes,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "packs": generated,
    }
    print(json.dumps(payload, indent=2))
    return 0


def _find_manifests(paths: Sequence[str]) -> list[Path]:
    found: list[Path] = []
    for raw in paths:
        path = Path(raw).resolve()
        if path.is_file() and path.name == "manifest.json":
            found.append(path)
        elif path.is_dir() and (path / "manifest.json").is_file():
            found.append(path / "manifest.json")
        elif path.is_dir():
            found.extend(sorted(path.rglob("manifest.json")))
    return sorted(set(found))


def _validate(args: argparse.Namespace) -> int:
    manifests = _find_manifests(args.paths)
    reports = [validate_pack(path) for path in manifests]
    payload = {
        "passed": bool(manifests) and all(report["passed"] for report in reports),
        "manifest_count": len(manifests),
        "reports": reports,
    }
    print(json.dumps(payload, indent=2))
    return 0 if payload["passed"] else 1


def _fuzz(args: argparse.Namespace) -> int:
    if args.no_isolation:
        report = fuzz_maps(
            args.count,
            base_seed=args.seed,
            width=args.width,
            height=args.height,
        )
    else:
        report = fuzz_maps_isolated(
            args.count,
            base_seed=args.seed,
            width=args.width,
            height=args.height,
            chunk_size=args.chunk_size,
            worker_retries=args.worker_retries,
            worker_timeout_seconds=args.worker_timeout,
        )
    if args.report:
        write_json_atomic(Path(args.report).resolve(), report)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


def _fuzz_worker(args: argparse.Namespace) -> int:
    if args.inject_exit_code:
        os._exit(args.inject_exit_code)
    report = fuzz_maps(
        args.count,
        base_seed=args.seed,
        width=args.width,
        height=args.height,
        start_index=args.start_index,
        include_hashes=True,
    )
    print(json.dumps(report, allow_nan=False))
    return 0 if report["passed"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deterministic offline semantic map forge.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate and atomically publish map packs.")
    format_major = MAP_SCHEMA_VERSION.split(".", 1)[0]
    generate.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR / f"maps_v{format_major}",
        help="Version-specific pack root (default: outputs/maps_v<schema-major>).",
    )
    generate.add_argument("--themes", nargs="+", default=["all"])
    generate.add_argument("--count", type=int, default=1, help="Maps per selected theme.")
    generate.add_argument("--seed", type=lambda value: int(value, 0), default=0x4E554C4C4D4150)
    generate.add_argument("--width", type=int, default=72)
    generate.add_argument("--height", type=int, default=72)
    generate.add_argument("--objectives", type=int, default=3)
    generate.add_argument("--spawns", type=int, default=12)
    generate.add_argument("--min-separation", type=int, default=0)
    generate.add_argument("--spawn-start-clearance", type=int, default=8)
    generate.add_argument("--spawn-objective-clearance", type=int, default=5)
    generate.add_argument("--spawn-hazard-clearance", type=int, default=2)
    generate.add_argument("--preview-scale", type=int, default=5)
    generate.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    generate.set_defaults(func=_generate)

    validate = subparsers.add_parser("validate", help="Validate one pack or recursively validate a directory.")
    validate.add_argument("paths", nargs="+", help="Pack directories, manifest files, or batch roots.")
    validate.set_defaults(func=_validate)

    fuzz = subparsers.add_parser("fuzz", help="Generate maps in memory and verify every invariant.")
    fuzz.add_argument("--count", type=int, default=500)
    fuzz.add_argument("--seed", type=lambda value: int(value, 0), default=0x46555A5A4D4150)
    fuzz.add_argument("--width", type=int, default=64)
    fuzz.add_argument("--height", type=int, default=64)
    fuzz.add_argument("--report", type=Path)
    fuzz.add_argument("--chunk-size", type=int, default=250)
    fuzz.add_argument("--worker-retries", type=int, default=2)
    fuzz.add_argument("--worker-timeout", type=int, default=300)
    fuzz.add_argument("--no-isolation", action="store_true")
    fuzz.set_defaults(func=_fuzz)

    worker = subparsers.add_parser("fuzz-worker", help=argparse.SUPPRESS)
    worker.add_argument("--count", type=int, required=True)
    worker.add_argument("--start-index", type=int, required=True)
    worker.add_argument("--seed", type=lambda value: int(value, 0), required=True)
    worker.add_argument("--width", type=int, required=True)
    worker.add_argument("--height", type=int, required=True)
    worker.add_argument("--inject-exit-code", type=int, default=0, help=argparse.SUPPRESS)
    worker.set_defaults(func=_fuzz_worker)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "count", 1) < 1:
        raise SystemExit("--count must be positive")
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
