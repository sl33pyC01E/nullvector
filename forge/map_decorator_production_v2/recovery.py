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
import uuid

from ..config import PROJECT_ROOT
from ..map_decorator.hashing import json_sha256
from ..map_decorator_ml.checkpoint import file_sha256
from ..map_decorator_production.corpus import (
    MANIFEST_FILE as CORPUS_MANIFEST_FILE,
    validate_corpus,
)
from ..safety import require_disk_floor
from .contract import (
    DISK_FLOOR_GIB,
    MAX_INDEX_WORKERS,
    MAX_PROCESS_ATTEMPTS,
    V2_CONTRACT_SHA256,
    V2_INDEX_FORMAT_VERSION,
)
from .index import (
    INDEX_MANIFEST_FILE,
    INDEX_VALIDATION_FILE,
    _aggregate,
    _atomic_json,
    _index_shard_path,
    _native_label,
    _source_manifest,
    _source_sha256,
    _validate_shard_index,
    validate_foreground_index,
)


RECOVERY_FORMAT_VERSION = "1.0.0"
RECOVERY_STATE_FILE = "foreground_index_recovery_state.json"
RECOVERY_REPORT_FILE = "foreground_index_power_recovery.json"
RECOVERY_TELEMETRY_FILE = "build_telemetry.json"
MAX_EVIDENCE_FILE_BYTES = 16 * 1024 * 1024


def _recovery_source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _tree_inventory(root: Path) -> dict[str, object]:
    root = Path(root).resolve()
    if not root.is_dir() or root.is_symlink():
        raise ValueError("Recovery evidence must be a real directory, not a link.")
    entries: list[dict[str, object]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink():
            raise ValueError(f"Recovery evidence contains a link: {path}")
        if not path.is_file():
            continue
        size = path.stat().st_size
        if size > MAX_EVIDENCE_FILE_BYTES:
            raise ValueError(f"Recovery evidence file exceeds its bound: {path}")
        entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": size,
                "sha256": file_sha256(path),
            }
        )
    return {
        "file_count": len(entries),
        "bytes": sum(int(entry["bytes"]) for entry in entries),
        "tree_sha256": json_sha256(entries),
        "entries": entries,
    }


def _scan_valid_shards(
    staging: Path,
    corpus_manifest: dict[str, object],
) -> dict[str, object]:
    staging = Path(staging).resolve()
    shards = corpus_manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("Corpus manifest has no foreground-index shards.")
    expected = {str(entry["shard_id"]): (index, entry) for index, entry in enumerate(shards)}
    shards_root = staging / "shards"
    crash_remnants: list[str] = []
    if shards_root.exists():
        if not shards_root.is_dir() or shards_root.is_symlink():
            raise ValueError("Foreground-index shard root is not a real directory.")
        for child in sorted(shards_root.iterdir(), key=lambda item: item.name):
            if child.is_symlink():
                raise ValueError(f"Foreground-index staging contains a shard link: {child}")
            if child.name.startswith("."):
                crash_remnants.append(child.relative_to(staging).as_posix())
                continue
            if child.name not in expected:
                raise ValueError(f"Foreground-index staging contains an unknown shard: {child.name}")
            if not child.is_dir():
                raise ValueError(f"Foreground-index shard entry is not a directory: {child}")
            members = sorted(item.name for item in child.iterdir())
            if members != ["counts.json"]:
                raise ValueError(f"Foreground-index shard directory is incomplete or unexpected: {child}")

    valid: list[dict[str, object]] = []
    missing: list[int] = []
    for shard_index, entry in enumerate(shards):
        path = _index_shard_path(staging, str(entry["shard_id"]))
        if not path.is_file():
            missing.append(shard_index)
            continue
        report = _validate_shard_index(
            path,
            corpus_sha256=str(corpus_manifest["corpus_sha256"]),
            entry=entry,
            shard_index=shard_index,
        )
        valid.append(
            {
                "shard_index": shard_index,
                "shard_id": entry["shard_id"],
                "file": path.relative_to(staging).as_posix(),
                "file_sha256": file_sha256(path),
                "samples_sha256": report["samples_sha256"],
                "sample_count": len(report["samples"]),
            }
        )
    return {
        "valid": valid,
        "missing": missing,
        "crash_remnants": crash_remnants,
        "valid_sample_count": sum(int(entry["sample_count"]) for entry in valid),
    }


def audit_foreground_staging(corpus: Path, staging: Path) -> dict[str, object]:
    corpus = Path(corpus).resolve()
    staging = Path(staging).resolve()
    corpus_validation = validate_corpus(corpus, verify_shards=False)
    corpus_manifest_path = corpus / CORPUS_MANIFEST_FILE
    corpus_manifest = json.loads(corpus_manifest_path.read_text(encoding="utf-8"))
    inventory = _tree_inventory(staging)
    scan = _scan_valid_shards(staging, corpus_manifest)
    return {
        "passed": True,
        "format_version": RECOVERY_FORMAT_VERSION,
        "staging": str(staging),
        "staging_file_count": inventory["file_count"],
        "staging_bytes": inventory["bytes"],
        "staging_tree_sha256": inventory["tree_sha256"],
        "corpus_sha256": corpus_manifest["corpus_sha256"],
        "corpus_manifest_sha256": file_sha256(corpus_manifest_path),
        "corpus_validation_passed": bool(corpus_validation["passed"]),
        "v2_contract_sha256": V2_CONTRACT_SHA256,
        "index_source_sha256": _source_sha256(),
        "index_source_manifest": _source_manifest(),
        "valid_shard_count": len(scan["valid"]),
        "valid_sample_count": scan["valid_sample_count"],
        "missing_shard_count": len(scan["missing"]),
        "missing_shard_indices": scan["missing"],
        "crash_remnants": scan["crash_remnants"],
        "valid_shards": scan["valid"],
    }


def _atomic_copy(source: Path, target: Path, *, expected_sha256: str) -> None:
    source = Path(source).resolve()
    target = Path(target)
    if target.exists():
        raise FileExistsError(f"Recovery copy target already exists: {target}")
    data = source.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError(f"Recovery source changed before copy: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    require_disk_floor(
        target.parent,
        floor_gb=DISK_FLOOR_GIB,
        planned_bytes=len(data) + 1024 * 1024,
    )
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.tmp-", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    if file_sha256(target) != expected_sha256:
        raise ValueError(f"Recovery copy failed byte-exact verification: {target}")


def _stop_active_workers(
    active: dict[subprocess.Popen[bytes], tuple[object, object]],
) -> None:
    for process in active:
        if process.poll() is None:
            process.terminate()
    for process, (stdout, stderr) in active.items():
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        stdout.close()  # type: ignore[union-attr]
        stderr.close()  # type: ignore[union-attr]


def _run_missing_workers(
    corpus: Path,
    output: Path,
    *,
    missing: list[int],
    python: Path,
    max_workers: int,
) -> list[dict[str, object]]:
    if not 1 <= max_workers <= MAX_INDEX_WORKERS:
        raise ValueError(f"Recovery workers must stay in [1,{MAX_INDEX_WORKERS}].")
    pending = deque((index, 1) for index in missing)
    active: dict[
        subprocess.Popen[bytes],
        tuple[int, int, object, object, float, Path, Path],
    ] = {}
    telemetry: list[dict[str, object]] = []
    logs = output / "telemetry"
    logs.mkdir(parents=True, exist_ok=True)
    try:
        while pending or active:
            while pending and len(active) < max_workers:
                shard_index, attempt = pending.popleft()
                require_disk_floor(
                    output,
                    floor_gb=DISK_FLOOR_GIB,
                    planned_bytes=64 * 1024 * 1024,
                )
                label = f"recovery-shard-{shard_index:03d}-attempt{attempt:02d}"
                stdout_path = logs / f"{label}.stdout.log"
                stderr_path = logs / f"{label}.stderr.log"
                stdout = stdout_path.open("xb")
                stderr = stderr_path.open("xb")
                try:
                    process = subprocess.Popen(
                        [
                            str(python),
                            "-m",
                            "forge.map_decorator_production_v2.index",
                            "worker",
                            "--corpus",
                            str(corpus),
                            "--output",
                            str(output),
                            "--shard-index",
                            str(shard_index),
                        ],
                        cwd=PROJECT_ROOT,
                        stdin=subprocess.DEVNULL,
                        stdout=stdout,
                        stderr=stderr,
                    )
                except BaseException:
                    stdout.close()
                    stderr.close()
                    raise
                active[process] = (
                    shard_index,
                    attempt,
                    stdout,
                    stderr,
                    time.perf_counter(),
                    stdout_path,
                    stderr_path,
                )
            completed = [process for process in active if process.poll() is not None]
            if not completed:
                time.sleep(0.10)
                continue
            for process in completed:
                (
                    shard_index,
                    attempt,
                    stdout,
                    stderr,
                    started,
                    stdout_path,
                    stderr_path,
                ) = active.pop(process)
                stdout.close()  # type: ignore[union-attr]
                stderr.close()  # type: ignore[union-attr]
                returncode = int(process.returncode or 0)
                record = {
                    "shard_index": shard_index,
                    "attempt": attempt,
                    "pid": process.pid,
                    "returncode": returncode,
                    "returncode_unsigned_hex": f"0x{returncode & 0xFFFF_FFFF:08x}",
                    "native_failure": _native_label(returncode),
                    "elapsed_seconds": time.perf_counter() - started,
                    "stdout": stdout_path.relative_to(output).as_posix(),
                    "stdout_sha256": file_sha256(stdout_path),
                    "stderr": stderr_path.relative_to(output).as_posix(),
                    "stderr_sha256": file_sha256(stderr_path),
                    "passed": returncode == 0,
                    "power_recovery": True,
                }
                telemetry.append(record)
                _atomic_json(output / RECOVERY_TELEMETRY_FILE, telemetry)
                if returncode != 0:
                    if attempt >= MAX_PROCESS_ATTEMPTS:
                        raise RuntimeError(
                            f"Foreground recovery shard {shard_index} exhausted its retry budget; "
                            f"last exit={record['returncode_unsigned_hex']}."
                        )
                    pending.append((shard_index, attempt + 1))
    except BaseException:
        _stop_active_workers(
            {process: (details[2], details[3]) for process, details in active.items()}
        )
        raise
    if not (output / RECOVERY_TELEMETRY_FILE).exists():
        _atomic_json(output / RECOVERY_TELEMETRY_FILE, telemetry)
    return telemetry


def recover_foreground_index(
    corpus: Path,
    source_staging: Path,
    output: Path,
    *,
    python: Path = Path(sys.executable),
    max_workers: int = MAX_INDEX_WORKERS,
) -> dict[str, object]:
    corpus = Path(corpus).resolve()
    source_staging = Path(source_staging).resolve()
    output = Path(output).resolve()
    python = Path(python).resolve()
    if output.exists():
        raise FileExistsError(f"Foreground recovery output already exists: {output}")
    if not python.is_file():
        raise FileNotFoundError(f"Foreground recovery Python executable is missing: {python}")
    if not 1 <= max_workers <= MAX_INDEX_WORKERS:
        raise ValueError(f"Recovery workers must stay in [1,{MAX_INDEX_WORKERS}].")
    if source_staging == output or source_staging in output.parents or output in source_staging.parents:
        raise ValueError("Recovery evidence and publication target must be disjoint.")
    output.parent.mkdir(parents=True, exist_ok=True)
    require_disk_floor(output.parent, floor_gb=DISK_FLOOR_GIB, planned_bytes=1024**3)

    audit = audit_foreground_staging(corpus, source_staging)
    inventory_before = _tree_inventory(source_staging)
    if audit["staging_tree_sha256"] != inventory_before["tree_sha256"]:
        raise RuntimeError("Recovery evidence changed during its initial audit.")
    corpus_manifest_path = corpus / CORPUS_MANIFEST_FILE
    corpus_manifest = json.loads(corpus_manifest_path.read_text(encoding="utf-8"))
    recovery_staging = output.parent / f".{output.name}.recovery-{uuid.uuid4().hex}"
    recovery_staging.mkdir(parents=True, exist_ok=False)

    imported = list(audit["valid_shards"])
    for entry in imported:
        source = source_staging / str(entry["file"])
        target = recovery_staging / str(entry["file"])
        _atomic_copy(source, target, expected_sha256=str(entry["file_sha256"]))

    state = {
        "format_version": RECOVERY_FORMAT_VERSION,
        "foreground_index_format_version": V2_INDEX_FORMAT_VERSION,
        "v2_contract_sha256": V2_CONTRACT_SHA256,
        "index_source_sha256": _source_sha256(),
        "index_source_manifest": _source_manifest(),
        "recovery_source_sha256": _recovery_source_sha256(),
        "corpus_sha256": corpus_manifest["corpus_sha256"],
        "corpus_manifest_sha256": file_sha256(corpus_manifest_path),
        "source_staging": str(source_staging),
        "source_staging_tree_sha256": inventory_before["tree_sha256"],
        "publication_target": str(output),
        "imported_shards": imported,
        "missing_shard_indices": audit["missing_shard_indices"],
        "max_workers": max_workers,
        "max_process_attempts": MAX_PROCESS_ATTEMPTS,
        "disk_floor_gib": DISK_FLOOR_GIB,
    }
    _atomic_json(recovery_staging / RECOVERY_STATE_FILE, state)

    missing = [int(index) for index in audit["missing_shard_indices"]]
    telemetry = _run_missing_workers(
        corpus,
        recovery_staging,
        missing=missing,
        python=python,
        max_workers=max_workers,
    )
    completed_scan = _scan_valid_shards(recovery_staging, corpus_manifest)
    if completed_scan["missing"]:
        raise RuntimeError(
            f"Foreground recovery still has missing shards: {completed_scan['missing']}"
        )
    manifest = _aggregate(corpus, recovery_staging, telemetry)

    inventory_after = _tree_inventory(source_staging)
    if inventory_after != inventory_before:
        raise RuntimeError("Recovery evidence changed while the additive rebuild was running.")
    validation = json.loads(
        (recovery_staging / INDEX_VALIDATION_FILE).read_text(encoding="utf-8")
    )
    report = {
        "passed": True,
        "format_version": RECOVERY_FORMAT_VERSION,
        "source_staging": str(source_staging),
        "source_staging_preserved": True,
        "source_staging_file_count": inventory_before["file_count"],
        "source_staging_bytes": inventory_before["bytes"],
        "source_staging_tree_sha256": inventory_before["tree_sha256"],
        "source_valid_shard_count": audit["valid_shard_count"],
        "source_valid_sample_count": audit["valid_sample_count"],
        "source_missing_shard_count": audit["missing_shard_count"],
        "source_missing_shard_indices": missing,
        "source_crash_remnants": audit["crash_remnants"],
        "imported_shard_count": len(imported),
        "built_shard_count": len(missing),
        "final_shard_count": len(completed_scan["valid"]),
        "final_sample_count": completed_scan["valid_sample_count"],
        "corpus_sha256": corpus_manifest["corpus_sha256"],
        "corpus_manifest_sha256": file_sha256(corpus_manifest_path),
        "v2_contract_sha256": V2_CONTRACT_SHA256,
        "index_source_sha256": _source_sha256(),
        "index_source_manifest": _source_manifest(),
        "recovery_source_sha256": _recovery_source_sha256(),
        "foreground_index_sha256": manifest["foreground_index_sha256"],
        "foreground_index_manifest_sha256": file_sha256(
            recovery_staging / INDEX_MANIFEST_FILE
        ),
        "foreground_index_validation_sha256": file_sha256(
            recovery_staging / INDEX_VALIDATION_FILE
        ),
        "validation": validation,
        "telemetry": {
            "worker_count": max_workers,
            "attempt_count": len(telemetry),
            "retry_count": sum(int(item["attempt"]) - 1 for item in telemetry),
            "native_failure_count": sum(item["native_failure"] is not None for item in telemetry),
            "windows_access_violation_count": sum(
                item["native_failure"] == "windows_access_violation" for item in telemetry
            ),
            "file": RECOVERY_TELEMETRY_FILE,
            "file_sha256": file_sha256(recovery_staging / RECOVERY_TELEMETRY_FILE),
            "max_process_attempts": MAX_PROCESS_ATTEMPTS,
        },
        "recovery_state": RECOVERY_STATE_FILE,
        "recovery_state_sha256": file_sha256(recovery_staging / RECOVERY_STATE_FILE),
        "publication_target": str(output),
        "atomic_publication": True,
        "disk_floor_gib": DISK_FLOOR_GIB,
    }
    _atomic_json(recovery_staging / RECOVERY_REPORT_FILE, report)
    staged_validation = validate_foreground_index(corpus, recovery_staging)
    if not staged_validation.get("passed"):
        raise RuntimeError("Recovered foreground index failed staged validation.")
    require_disk_floor(output.parent, floor_gb=DISK_FLOOR_GIB, planned_bytes=1024**3)
    if output.exists():
        raise FileExistsError(f"Foreground recovery output appeared before publish: {output}")
    os.replace(recovery_staging, output)
    published_validation = validate_foreground_index(corpus, output)
    if not published_validation.get("passed"):
        raise RuntimeError("Published foreground index failed validation.")
    final_inventory = _tree_inventory(source_staging)
    if final_inventory != inventory_before:
        raise RuntimeError("Recovery evidence changed across atomic publication.")
    return {
        "passed": True,
        "output": str(output),
        "foreground_index_sha256": manifest["foreground_index_sha256"],
        "manifest_sha256": file_sha256(output / INDEX_MANIFEST_FILE),
        "validation_sha256": file_sha256(output / INDEX_VALIDATION_FILE),
        "recovery_report_sha256": file_sha256(output / RECOVERY_REPORT_FILE),
        "source_staging_preserved": True,
        "source_staging_tree_sha256": inventory_before["tree_sha256"],
        "imported_shard_count": len(imported),
        "built_shard_count": len(missing),
        "final_shard_count": len(completed_scan["valid"]),
        "final_sample_count": completed_scan["valid_sample_count"],
        "telemetry": report["telemetry"],
        "validation": published_validation,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Additive, evidence-preserving recovery for a v2 foreground index"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--corpus", type=Path, required=True)
    audit.add_argument("--staging", type=Path, required=True)
    recover = subparsers.add_parser("recover")
    recover.add_argument("--corpus", type=Path, required=True)
    recover.add_argument("--source-staging", type=Path, required=True)
    recover.add_argument("--output", type=Path, required=True)
    recover.add_argument("--python", type=Path, default=Path(sys.executable))
    recover.add_argument("--workers", type=int, default=MAX_INDEX_WORKERS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "audit":
        report = audit_foreground_staging(args.corpus, args.staging)
    else:
        report = recover_foreground_index(
            args.corpus,
            args.source_staging,
            args.output,
            python=args.python,
            max_workers=args.workers,
        )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
