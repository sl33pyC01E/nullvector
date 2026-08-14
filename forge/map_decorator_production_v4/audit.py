from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Final
import uuid

from ..map_decorator.hashing import json_sha256
from ..safety import require_disk_floor
from .proposal import ProposalAuthority, audit_proposal_targets
from .smoke import source_sha256


AUDIT_FORMAT: Final[str] = "nullvector-map-decorator-v4-full-corpus-proposal-audit/1.0.0"
CHUNK_FORMAT: Final[str] = "nullvector-map-decorator-v4-proposal-audit-chunk/1.0.0"
AUDIT_REPORT_NAME: Final[str] = "proposal_audit_report.json"
TOTAL_SHARDS: Final[int] = 216
SHARDS_PER_CHUNK: Final[int] = 12
CHUNK_COUNT: Final[int] = TOTAL_SHARDS // SHARDS_PER_CHUNK
MAX_ATTEMPTS: Final[int] = 3
MAX_WORKERS: Final[int] = 2


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    require_disk_floor(path.parent, floor_gb=100.0, planned_bytes=len(encoded) + 1024 * 1024)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _read_hashed_json(path: Path, *, maximum_bytes: int = 64 * 1024 * 1024) -> dict[str, Any]:
    path = Path(path).resolve()
    if not path.is_file() or path.is_symlink() or not 0 < path.stat().st_size <= maximum_bytes:
        raise ValueError("V4 audit JSON is missing, unsafe, or oversized.")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("V4 audit JSON root must be an object.")
    stored = value.pop("report_sha256", None)
    if stored != json_sha256(value):
        raise ValueError("V4 audit JSON self-hash failed.")
    value["report_sha256"] = stored
    return value


def _empty_counts() -> dict[str, dict[str, int]]:
    return {
        split: {
            f"{head}_{metric}": 0
            for head in ("decal", "prop")
            for metric in ("target", "proposal", "hit", "missing", "extra")
        }
        for split in ("train", "validation", "test")
    }


def audit_chunk(
    corpus_root: Path,
    index_root: Path,
    output: Path,
    *,
    chunk_index: int,
) -> dict[str, Any]:
    if isinstance(chunk_index, bool) or not 0 <= chunk_index < CHUNK_COUNT:
        raise ValueError("V4 audit chunk index is outside its bounded contract.")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("V4 audit chunk output is immutable.")
    authority = ProposalAuthority.load(Path(corpus_root), Path(index_root))
    start_shard = chunk_index * SHARDS_PER_CHUNK
    stop_shard = start_shard + SHARDS_PER_CHUNK
    refs = sorted(
        (
            ref
            for split_refs in authority.authority.corpus.refs_by_split.values()
            for ref in split_refs
            if start_shard <= ref.shard_index < stop_shard
        ),
        key=lambda ref: (ref.shard_index, ref.sample_index),
    )
    records: list[dict[str, object]] = []
    counts = _empty_counts()
    themes: dict[str, int] = {}
    for ref in refs:
        sample, proposals = authority.sample_and_proposals(ref)
        audit = audit_proposal_targets(proposals, sample.targets)
        if not audit["passed"]:
            raise RuntimeError("V4 full-corpus proposal audit missed a target object cell.")
        theme = proposals.theme
        themes[theme] = themes.get(theme, 0) + 1
        for head in ("decal", "prop"):
            item = audit["heads"][head]
            for metric in ("target", "proposal", "hit", "missing", "extra"):
                counts[ref.split][f"{head}_{metric}"] += int(item[f"{metric}_count"])
        records.append(
            {
                "shard_index": ref.shard_index,
                "sample_index": ref.sample_index,
                "split": ref.split,
                "theme": theme,
                "sample_identity_sha256": ref.sample_identity_sha256,
                "full_map_identity_sha256": ref.full_map_identity_sha256,
                "proposal_fields_sha256": proposals.fields_sha256,
                "heads": audit["heads"],
            }
        )
    report: dict[str, object] = {
        "format": CHUNK_FORMAT,
        "status": "passed",
        "source_sha256": source_sha256(),
        "authority": {
            "corpus_sha256": authority.authority.corpus.corpus_sha256,
            "index_semantic_sha256": authority.authority.index_semantic_sha256,
        },
        "chunk_index": chunk_index,
        "start_shard": start_shard,
        "stop_shard_exclusive": stop_shard,
        "sample_count": len(records),
        "themes": dict(sorted(themes.items())),
        "counts": counts,
        "records_sha256": json_sha256(records),
        "records": records,
        "gates": {
            "exact_shard_range": {record["shard_index"] for record in records} == set(range(start_shard, stop_shard)),
            "no_duplicate_samples": len({record["sample_identity_sha256"] for record in records}) == len(records),
            "zero_missing_target_cells": all(
                counts[split][f"{head}_missing"] == 0
                for split in counts
                for head in ("decal", "prop")
            ),
        },
    }
    if not all(report["gates"].values()):  # type: ignore[union-attr]
        raise RuntimeError("V4 audit chunk gate failed.")
    report["report_sha256"] = json_sha256(report)
    _atomic_json(output, report)
    return _read_hashed_json(output)


def _exit_class(returncode: int) -> str:
    unsigned = returncode & 0xFFFFFFFF
    if unsigned == 0xC0000005:
        return "windows_access_violation"
    return "success" if returncode == 0 else f"exit_{unsigned:08x}"


def _run_chunk_process(
    corpus_root: Path,
    index_root: Path,
    output: Path,
    chunk_index: int,
    attempt: int,
    telemetry_root: Path,
) -> dict[str, object]:
    command = [
        sys.executable,
        "-m",
        "forge.map_decorator_production_v4",
        "audit-worker",
        "--corpus",
        str(Path(corpus_root).resolve()),
        "--index",
        str(Path(index_root).resolve()),
        "--output",
        str(output),
        "--chunk-index",
        str(chunk_index),
    ]
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
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[2],
        env=environment,
        capture_output=True,
        timeout=900,
        check=False,
    )
    stdout = completed.stdout[-1024 * 1024 :]
    stderr = completed.stderr[-1024 * 1024 :]
    telemetry_root.mkdir(parents=True, exist_ok=True)
    prefix = telemetry_root / f"chunk_{chunk_index:02d}_attempt_{attempt}"
    prefix.with_suffix(".stdout.log").write_bytes(stdout)
    prefix.with_suffix(".stderr.log").write_bytes(stderr)
    return {
        "chunk_index": chunk_index,
        "attempt": attempt,
        "returncode": completed.returncode,
        "exit_class": _exit_class(completed.returncode),
        "elapsed_seconds": time.perf_counter() - started,
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
    }


def _aggregate_chunks(chunks: list[dict[str, Any]]) -> tuple[list[dict[str, object]], dict[str, dict[str, int]]]:
    records: list[dict[str, object]] = []
    counts = _empty_counts()
    for chunk in sorted(chunks, key=lambda item: int(item["chunk_index"])):
        records.extend(chunk["records"])
        for split in counts:
            for key in counts[split]:
                counts[split][key] += int(chunk["counts"][split][key])
    return records, counts


def build_full_audit(
    corpus_root: Path,
    index_root: Path,
    output: Path,
) -> dict[str, Any]:
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("V4 full proposal audit output is immutable.")
    require_disk_floor(output.parent, floor_gb=100.0, planned_bytes=256 * 1024 * 1024)
    staging = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    chunks_root = staging / "chunks"
    telemetry_root = staging / "telemetry"
    chunks_root.mkdir(parents=True)
    pending = deque((index, 1) for index in range(CHUNK_COUNT))
    telemetry: list[dict[str, object]] = []
    while pending:
        wave = [pending.popleft() for _ in range(min(MAX_WORKERS, len(pending) or 1))]
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(
                    _run_chunk_process,
                    corpus_root,
                    index_root,
                    chunks_root / f"chunk_{index:02d}.json",
                    index,
                    attempt,
                    telemetry_root,
                ): (index, attempt)
                for index, attempt in wave
            }
            for future in as_completed(futures):
                index, attempt = futures[future]
                try:
                    record = future.result()
                except subprocess.TimeoutExpired:
                    record = {
                        "chunk_index": index,
                        "attempt": attempt,
                        "returncode": None,
                        "exit_class": "timeout",
                        "elapsed_seconds": 900.0,
                    }
                telemetry.append(record)
                if record["exit_class"] == "success":
                    _read_hashed_json(chunks_root / f"chunk_{index:02d}.json")
                elif attempt < MAX_ATTEMPTS:
                    pending.append((index, attempt + 1))
                else:
                    failed = output.parent / f"{output.name}.failed-{uuid.uuid4().hex}"
                    _atomic_json(staging / "failure.json", {"format": AUDIT_FORMAT, "telemetry": telemetry})
                    os.replace(staging, failed)
                    raise RuntimeError(f"V4 proposal audit chunk {index} exhausted retries: {failed}")
    chunks = [_read_hashed_json(chunks_root / f"chunk_{index:02d}.json") for index in range(CHUNK_COUNT)]
    records, counts = _aggregate_chunks(chunks)
    authority = ProposalAuthority.load(Path(corpus_root), Path(index_root))
    expected_ids = sorted(
        ref.sample_identity_sha256
        for refs in authority.authority.corpus.refs_by_split.values()
        for ref in refs
    )
    observed_ids = sorted(str(record["sample_identity_sha256"]) for record in records)
    summary: dict[str, object] = {}
    for split in counts:
        split_summary: dict[str, object] = {}
        for head in ("decal", "prop"):
            target = counts[split][f"{head}_target"]
            proposal = counts[split][f"{head}_proposal"]
            hit = counts[split][f"{head}_hit"]
            split_summary[head] = {
                "target_count": target,
                "proposal_count": proposal,
                "hit_count": hit,
                "missing_count": counts[split][f"{head}_missing"],
                "extra_count": counts[split][f"{head}_extra"],
                "recall": 1.0 if target == 0 else hit / target,
                "precision": 1.0 if proposal == 0 else hit / proposal,
            }
        summary[split] = split_summary
    report: dict[str, object] = {
        "format": AUDIT_FORMAT,
        "status": "passed",
        "source_sha256": source_sha256(),
        "authority": {
            "corpus_sha256": authority.authority.corpus.corpus_sha256,
            "corpus_manifest_sha256": authority.authority.corpus.manifest_sha256,
            "index_semantic_sha256": authority.authority.index_semantic_sha256,
            "index_manifest_sha256": authority.authority.index_manifest_sha256,
        },
        "counts": {
            "chunk_count": CHUNK_COUNT,
            "source_shard_count": TOTAL_SHARDS,
            "sample_count": len(records),
            "attempt_count": len(telemetry),
            "retry_count": len(telemetry) - CHUNK_COUNT,
        },
        "sample_identity_sha256": json_sha256(observed_ids),
        "proposal_record_sha256": json_sha256(records),
        "summary": summary,
        "chunks": [
            {
                "path": f"chunks/chunk_{index:02d}.json",
                "report_sha256": chunks[index]["report_sha256"],
                "records_sha256": chunks[index]["records_sha256"],
                "sample_count": chunks[index]["sample_count"],
            }
            for index in range(CHUNK_COUNT)
        ],
        "telemetry": sorted(telemetry, key=lambda item: (int(item["chunk_index"]), int(item["attempt"]))),
        "gates": {
            "all_216_source_shards": {record["shard_index"] for record in records} == set(range(TOTAL_SHARDS)),
            "all_3096_samples": len(records) == 3_096,
            "exact_authoritative_sample_registry": observed_ids == expected_ids,
            "zero_duplicate_samples": len(observed_ids) == len(set(observed_ids)),
            "zero_missing_target_cells": all(
                counts[split][f"{head}_missing"] == 0
                for split in counts
                for head in ("decal", "prop")
            ),
            "bounded_process_attempts": all(int(item["attempt"]) <= MAX_ATTEMPTS for item in telemetry),
            "every_chunk_succeeded": all(item["status"] == "passed" for item in chunks),
            "cpu_only": True,
            "not_a_quality_claim": True,
        },
    }
    if not all(report["gates"].values()):  # type: ignore[union-attr]
        raise RuntimeError(f"V4 full proposal audit failed: {report['gates']}")
    report["report_sha256"] = json_sha256(report)
    _atomic_json(staging / AUDIT_REPORT_NAME, report)
    os.replace(staging, output)
    return validate_full_audit(output, corpus_root=corpus_root, index_root=index_root)


def validate_full_audit(
    output: Path,
    *,
    corpus_root: Path,
    index_root: Path,
) -> dict[str, Any]:
    output = Path(output).resolve()
    report = _read_hashed_json(output / AUDIT_REPORT_NAME)
    if report.get("format") != AUDIT_FORMAT or report.get("status") != "passed" or report.get("source_sha256") != source_sha256():
        raise ValueError("V4 full proposal audit format/source/status failed.")
    chunks = []
    for index, artifact in enumerate(report["chunks"]):
        if artifact.get("path") != f"chunks/chunk_{index:02d}.json":
            raise ValueError("V4 audit chunk path registry is noncanonical.")
        chunk = _read_hashed_json(output / artifact["path"])
        if chunk["report_sha256"] != artifact["report_sha256"] or chunk["records_sha256"] != artifact["records_sha256"] or chunk["sample_count"] != artifact["sample_count"]:
            raise ValueError("V4 audit chunk artifact closure failed.")
        chunks.append(chunk)
    records, counts = _aggregate_chunks(chunks)
    authority = ProposalAuthority.load(Path(corpus_root), Path(index_root))
    expected_ids = sorted(
        ref.sample_identity_sha256
        for refs in authority.authority.corpus.refs_by_split.values()
        for ref in refs
    )
    observed_ids = sorted(str(record["sample_identity_sha256"]) for record in records)
    if report["sample_identity_sha256"] != json_sha256(observed_ids) or observed_ids != expected_ids:
        raise ValueError("V4 audit sample registry failed replay.")
    if report["proposal_record_sha256"] != json_sha256(records):
        raise ValueError("V4 audit proposal record identity failed replay.")
    for split in counts:
        for head in ("decal", "prop"):
            item = report["summary"][split][head]
            expected = {
                "target_count": counts[split][f"{head}_target"],
                "proposal_count": counts[split][f"{head}_proposal"],
                "hit_count": counts[split][f"{head}_hit"],
                "missing_count": counts[split][f"{head}_missing"],
                "extra_count": counts[split][f"{head}_extra"],
            }
            for key, value in expected.items():
                if item[key] != value:
                    raise ValueError("V4 audit summary failed exact aggregation replay.")
    if not all(report["gates"].values()):
        raise ValueError("V4 audit recorded a failed gate.")
    return report
