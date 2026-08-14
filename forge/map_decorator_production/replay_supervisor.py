from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable

from ..config import PROJECT_ROOT
from ..map_decorator.hashing import json_sha256
from ..map_decorator_ml.checkpoint import file_sha256
from ..safety import require_disk_floor
from .contract import DISK_FLOOR_GIB, MAX_PROCESS_ATTEMPTS, MAX_WORKERS
from .corpus import MANIFEST_FILE, VALIDATION_FILE, ShardSpec, validate_corpus
from .provenance import source_sha256


FORMAT = "nullvector-map-decorator-corpus-isolated-replay-v1"
RESULT_FORMAT = "nullvector-map-decorator-corpus-isolated-shard-result-v1"
SOURCE_FILES = (
    "forge/map_decorator_production/replay_supervisor.py",
    "forge/map_decorator_production/worker.py",
)
THREAD_ENV = {
    "CUDA_VISIBLE_DEVICES": "-1",
    "PYTHONHASHSEED": "0",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def replay_source_sha256() -> str:
    digest = hashlib.sha256(b"nullvector-map-decorator-corpus-replay-source-v1\0")
    digest.update(source_sha256("corpus").encode("ascii"))
    digest.update(b"\0")
    for relative in SOURCE_FILES:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        digest.update(relative.encode("utf-8")); digest.update(b"\0")
        digest.update(path.read_bytes()); digest.update(b"\0")
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _atomic_json(path: Path, value: Any) -> None:
    payload = _canonical_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def _native_label(returncode: int) -> str | None:
    return {
        0xC0000005: "windows_access_violation",
        0xC0000409: "windows_stack_buffer_overrun",
        0xC000001D: "windows_illegal_instruction",
    }.get(returncode & 0xFFFF_FFFF)


def _safe_relative(root: Path, relative: str) -> Path:
    if not relative or "\\" in relative or relative.startswith(("/", "./")):
        raise ValueError("Replay artifact path is unsafe.")
    pieces = relative.split("/")
    if any(piece in {"", ".", ".."} for piece in pieces):
        raise ValueError("Replay artifact path contains an unsafe segment.")
    root = root.resolve(); path = root.joinpath(*pieces).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Replay artifact escapes its root.")
    return path


def _load_attempts(output: Path, shard_id: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted((output / "attempts").glob(f"{shard_id}-attempt*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("shard_id") != shard_id:
            raise ValueError(f"Malformed attempt telemetry: {path}")
        stored_hash = value.pop("attempt_sha256", None)
        if stored_hash != json_sha256(value):
            raise ValueError(f"Attempt telemetry hash failed: {path}")
        value["attempt_sha256"] = stored_hash
        if value.get("format") != FORMAT:
            raise ValueError(f"Attempt telemetry format failed: {path}")
        stdout = _safe_relative(output, str(value.get("stdout", "")))
        stderr = _safe_relative(output, str(value.get("stderr", "")))
        if not stdout.is_file() or file_sha256(stdout) != value.get("stdout_sha256"):
            raise ValueError(f"Attempt stdout is missing or changed: {path}")
        if not stderr.is_file() or file_sha256(stderr) != value.get("stderr_sha256"):
            raise ValueError(f"Attempt stderr is missing or changed: {path}")
        records.append(value)
    expected = list(range(1, len(records) + 1))
    if [int(record.get("attempt", -1)) for record in records] != expected:
        raise ValueError(f"Attempt telemetry is not contiguous for {shard_id}.")
    return records


def _validate_result(
    path: Path,
    *,
    output: Path,
    spec: ShardSpec,
    corpus_sha256: str,
    replay_source: str,
) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    expected_keys = {
        "format", "status", "shard_id", "sample_count", "corpus_sha256",
        "corpus_source_sha256", "replay_source_sha256", "attempt", "stdout",
        "stdout_sha256", "stderr", "stderr_sha256", "worker_report",
        "result_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ValueError(f"Shard replay result members are malformed: {path}")
    result_hash = value.pop("result_sha256")
    if result_hash != json_sha256(value):
        raise ValueError(f"Shard replay result hash failed: {path}")
    value["result_sha256"] = result_hash
    if value["format"] != RESULT_FORMAT or value["status"] != "passed":
        raise ValueError(f"Shard replay result is not passed: {path}")
    expected = {
        "shard_id": spec.shard_id,
        "sample_count": spec.sample_count,
        "corpus_sha256": corpus_sha256,
        "corpus_source_sha256": source_sha256("corpus"),
        "replay_source_sha256": replay_source,
    }
    for key, target in expected.items():
        if value.get(key) != target:
            raise ValueError(f"Shard replay result {key} is stale: {path}")
    stdout = _safe_relative(output, str(value["stdout"])); stderr = _safe_relative(output, str(value["stderr"]))
    if not stdout.is_file() or file_sha256(stdout) != value["stdout_sha256"]:
        raise ValueError(f"Shard replay stdout is missing or changed: {path}")
    if not stderr.is_file() or file_sha256(stderr) != value["stderr_sha256"]:
        raise ValueError(f"Shard replay stderr is missing or changed: {path}")
    report = value.get("worker_report")
    if not isinstance(report, dict) or report.get("passed") is not True:
        raise ValueError(f"Shard replay worker report is malformed: {path}")
    if report.get("shard_id") != spec.shard_id or report.get("sample_count") != spec.sample_count:
        raise ValueError(f"Shard replay worker report identity drifted: {path}")
    if report.get("exact_semantic_feature_target_legality_replay") is not True:
        raise ValueError(f"Shard replay was not exact: {path}")
    return value


def _production_specs(corpus: Path) -> tuple[str, tuple[ShardSpec, ...], dict[str, str]]:
    base = validate_corpus(corpus, verify_shards=False)
    manifest_path = corpus / MANIFEST_FILE; validation_path = corpus / VALIDATION_FILE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    specs = tuple(ShardSpec.from_dict(entry["spec"]) for entry in manifest["shards"])
    if len(specs) != len({spec.shard_id for spec in specs}):
        raise ValueError("Production corpus has duplicate shard identities.")
    for spec in specs:
        if not (corpus / spec.validation_path).is_file():
            raise FileNotFoundError(f"Sealed shard validation is missing: {spec.shard_id}")
    return str(base["corpus_sha256"]), specs, {
        "manifest": file_sha256(manifest_path),
        "validation": file_sha256(validation_path),
    }


def run_isolated_replay(
    corpus: Path,
    output: Path,
    *,
    python: Path = Path(sys.executable),
    workers: int = MAX_WORKERS,
    max_attempts: int = MAX_PROCESS_ATTEMPTS,
    timeout_seconds: int = 900,
    specs: Iterable[ShardSpec] | None = None,
    fixture_corpus_sha256: str | None = None,
) -> dict[str, Any]:
    corpus = Path(corpus).resolve(); output = Path(output).resolve(); python = Path(python).resolve()
    if not 1 <= workers <= MAX_WORKERS or not 1 <= max_attempts <= MAX_PROCESS_ATTEMPTS:
        raise ValueError("Replay workers/attempts exceed the bounded contract.")
    if timeout_seconds < 30:
        raise ValueError("Replay timeout is unreasonably short.")
    output.mkdir(parents=True, exist_ok=True)
    if specs is None:
        corpus_hash, selected_specs, corpus_files_before = _production_specs(corpus)
    else:
        selected_specs = tuple(specs)
        if not fixture_corpus_sha256 or len(fixture_corpus_sha256) != 64:
            raise ValueError("Fixture replay requires an explicit 64-character corpus identity.")
        corpus_hash = fixture_corpus_sha256; corpus_files_before = {}
    replay_source = replay_source_sha256()
    final_path = output / "replay_report.json"
    if final_path.exists():
        report = json.loads(final_path.read_text(encoding="utf-8"))
        stored_hash = report.pop("replay_sha256", None)
        if stored_hash != json_sha256(report):
            raise ValueError("Existing replay report self-hash failed.")
        report["replay_sha256"] = stored_hash
        if report.get("status") != "passed" or report.get("replay_source_sha256") != replay_source or report.get("corpus_sha256") != corpus_hash:
            raise ValueError("Existing replay report is stale or failed.")
        results = {
            spec.shard_id: _validate_result(output / "results" / f"{spec.shard_id}.json", output=output, spec=spec, corpus_sha256=corpus_hash, replay_source=replay_source)
            for spec in selected_specs
        }
        attempts = [record for spec in selected_specs for record in _load_attempts(output, spec.shard_id)]
        expected_counts = {"shards": len(selected_specs), "samples": sum(spec.sample_count for spec in selected_specs), "attempts": len(attempts), "retries": len(attempts) - len(selected_specs), "native_failures": sum(bool(record.get("native_failure")) for record in attempts)}
        if report.get("counts") != expected_counts or report.get("shard_result_sha256") != [results[spec.shard_id]["result_sha256"] for spec in selected_specs]:
            raise ValueError("Existing replay report no longer closes over its artifacts.")
        if not isinstance(report.get("gates"), dict) or not all(report["gates"].values()):
            raise ValueError("Existing replay report has a failed gate.")
        return report

    (output / "logs").mkdir(exist_ok=True); (output / "attempts").mkdir(exist_ok=True); (output / "results").mkdir(exist_ok=True)
    pending: deque[ShardSpec] = deque()
    results: dict[str, dict[str, Any]] = {}
    for spec in selected_specs:
        result_path = output / "results" / f"{spec.shard_id}.json"
        if result_path.exists():
            results[spec.shard_id] = _validate_result(result_path, output=output, spec=spec, corpus_sha256=corpus_hash, replay_source=replay_source)
        else:
            pending.append(spec)

    active: dict[subprocess.Popen[bytes], tuple[ShardSpec, int, Any, Any, float, Path, Path]] = {}
    environment = os.environ.copy(); environment.update(THREAD_ENV)
    while pending or active:
        while pending and len(active) < workers:
            spec = pending.popleft(); prior = _load_attempts(output, spec.shard_id); attempt = len(prior) + 1
            if attempt > max_attempts:
                raise RuntimeError(f"Shard {spec.shard_id} exhausted {max_attempts} attempts.")
            require_disk_floor(output, floor_gb=DISK_FLOOR_GIB, planned_bytes=64 * 1024 * 1024)
            stdout = output / "logs" / f"{spec.shard_id}-attempt{attempt:02d}.stdout.log"
            stderr = output / "logs" / f"{spec.shard_id}-attempt{attempt:02d}.stderr.log"
            stdout_handle = stdout.open("xb"); stderr_handle = stderr.open("xb")
            command = [str(python), "-m", "forge.map_decorator_production.worker", "validate", "--spec", str(corpus / "specs" / f"{spec.shard_id}.json"), "--root", str(corpus)]
            process = subprocess.Popen(command, cwd=PROJECT_ROOT, stdin=subprocess.DEVNULL, stdout=stdout_handle, stderr=stderr_handle, env=environment)
            active[process] = (spec, attempt, stdout_handle, stderr_handle, time.perf_counter(), stdout, stderr)
        completed = [process for process, state in active.items() if process.poll() is not None or time.perf_counter() - state[4] > timeout_seconds]
        if not completed:
            time.sleep(.1); continue
        for process in completed:
            spec, attempt, stdout_handle, stderr_handle, started, stdout, stderr = active.pop(process)
            timed_out = process.poll() is None
            if timed_out:
                process.kill(); process.wait(timeout=10)
            stdout_handle.close(); stderr_handle.close()
            returncode = int(process.returncode or 0) if not timed_out else -9
            attempt_record = {
                "format": FORMAT, "shard_id": spec.shard_id, "attempt": attempt,
                "returncode": returncode, "returncode_unsigned_hex": f"0x{returncode & 0xFFFFFFFF:08x}",
                "native_failure": "timeout" if timed_out else _native_label(returncode),
                "elapsed_seconds": time.perf_counter() - started,
                "stdout": stdout.relative_to(output).as_posix(), "stdout_sha256": file_sha256(stdout),
                "stderr": stderr.relative_to(output).as_posix(), "stderr_sha256": file_sha256(stderr),
                "passed": returncode == 0,
            }
            attempt_record["attempt_sha256"] = json_sha256(attempt_record)
            _atomic_json(output / "attempts" / f"{spec.shard_id}-attempt{attempt:02d}.json", attempt_record)
            if returncode != 0:
                if attempt >= max_attempts:
                    raise RuntimeError(f"Shard {spec.shard_id} failed after {attempt} attempts ({attempt_record['returncode_unsigned_hex']}).")
                pending.append(spec); continue
            try:
                worker_report = json.loads(stdout.read_text(encoding="utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                if attempt >= max_attempts:
                    raise RuntimeError(f"Shard {spec.shard_id} emitted malformed success output.") from error
                pending.append(spec); continue
            result = {
                "format": RESULT_FORMAT, "status": "passed", "shard_id": spec.shard_id,
                "sample_count": spec.sample_count, "corpus_sha256": corpus_hash,
                "corpus_source_sha256": source_sha256("corpus"), "replay_source_sha256": replay_source,
                "attempt": attempt, "stdout": stdout.relative_to(output).as_posix(), "stdout_sha256": file_sha256(stdout),
                "stderr": stderr.relative_to(output).as_posix(), "stderr_sha256": file_sha256(stderr),
                "worker_report": worker_report,
            }
            result["result_sha256"] = json_sha256(result)
            result_path = output / "results" / f"{spec.shard_id}.json"; _atomic_json(result_path, result)
            results[spec.shard_id] = _validate_result(result_path, output=output, spec=spec, corpus_sha256=corpus_hash, replay_source=replay_source)

    if set(results) != {spec.shard_id for spec in selected_specs}:
        raise RuntimeError("Isolated replay did not cover the exact shard set.")
    attempts = [record for spec in selected_specs for record in _load_attempts(output, spec.shard_id)]
    corpus_files_after = corpus_files_before
    if specs is None:
        _, _, corpus_files_after = _production_specs(corpus)
        if corpus_files_after != corpus_files_before:
            raise ValueError("Sealed corpus authority files changed during replay.")
    base_report = {
        "format": FORMAT, "status": "passed", "mode": "production" if specs is None else "fixture",
        "corpus_sha256": corpus_hash, "corpus_source_sha256": source_sha256("corpus"),
        "replay_source_sha256": replay_source, "corpus_authority_files": corpus_files_after,
        "counts": {"shards": len(selected_specs), "samples": sum(spec.sample_count for spec in selected_specs), "attempts": len(attempts), "retries": len(attempts) - len(selected_specs), "native_failures": sum(bool(record.get("native_failure")) for record in attempts)},
        "shard_result_sha256": [results[spec.shard_id]["result_sha256"] for spec in selected_specs],
        "gates": {"all_shards_exact_replay": True, "all_samples_exact_replay": True, "bounded_process_isolation": True, "corpus_source_hash_exact": True, "corpus_authority_unchanged": corpus_files_after == corpus_files_before, "disk_floor_preserved": True, "no_cuda": True},
    }
    base_report["replay_sha256"] = json_sha256(base_report); _atomic_json(final_path, base_report)
    return base_report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resumable process-isolated map-decorator corpus replay")
    parser.add_argument("--corpus", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable)); parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--max-attempts", type=int, default=MAX_PROCESS_ATTEMPTS); parser.add_argument("--timeout-seconds", type=int, default=900)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_isolated_replay(args.corpus, args.output, python=args.python, workers=args.workers, max_attempts=args.max_attempts, timeout_seconds=args.timeout_seconds)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
