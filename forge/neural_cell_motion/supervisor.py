from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any
import uuid

from ..config import PROJECT_ROOT
from ..multifield_style_motion.hashing import canonical_json_bytes
from ..safety import require_disk_floor
from .contract import DEFAULT_ANATOMY, DEFAULT_CORPUS, DEFAULT_MOTION, corpus_source_sha256
from .dataset import _atomic_bytes, _read_canonical_json, _selection_plan, _write_corpus_manifest, sha256_bytes, sha256_file
from .worker import PREFLIGHT_FORMAT, RESULT_FORMAT


SUPERVISOR_FORMAT = "nullvector-neural-cell-motion-resilient-build-v1"
VALIDATION_FORMAT = "nullvector-neural-cell-motion-resilient-validation-v1"
ACCESS_VIOLATION_CODES = {0xC0000005, -1073741819, 0xC0000409, -1073740791}


def _worker_environment() -> dict[str, str]:
    result = os.environ.copy()
    result.update({"CUDA_VISIBLE_DEVICES": "-1", "PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"})
    return result


def _write_spec(path: Path, record: dict[str, Any], ordinal: int, split: str, identities_per_family: int | None, motion_sha: str, anatomy_sha: str) -> dict[str, Any]:
    spec: dict[str, Any] = {"format": "nullvector-neural-cell-motion-worker-spec-v1", "source_sha256": corpus_source_sha256(), "identities_per_family": identities_per_family, "sample_id": record["sample_id"], "family": record["family"], "family_id": int(record["family_id"]), "family_ordinal": ordinal, "split": split, "motion_manifest_sha256": motion_sha, "anatomy_manifest_sha256": anatomy_sha}
    spec["semantic_sha256"] = sha256_bytes(canonical_json_bytes(spec)); _atomic_bytes(path, canonical_json_bytes(spec)); return spec


def _validate_records_isolated(output: Path, sample_ids: list[str], *, replay: bool, workers: int, max_attempts: int, timeout_seconds: int) -> list[dict[str, Any]]:
    jobs = [{"sample_id": sample_id, "attempt": 0} for sample_id in sample_ids]
    active: dict[subprocess.Popen[str], dict[str, Any]] = {}; events: list[dict[str, Any]] = []
    manifest = _read_canonical_json(output / "neural_cell_motion_corpus.json")
    while jobs or active:
        while jobs and len(active) < workers:
            job = jobs.pop(0); job["attempt"] += 1
            command = [sys.executable, "-m", "forge.neural_cell_motion", "validate-corpus", "--output", str(output), "--sample-id", job["sample_id"]]
            if replay: command.append("--replay")
            process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=_worker_environment(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            job["started"] = time.monotonic(); active[process] = job
        if not active: break
        time.sleep(.1)
        for process, job in list(active.items()):
            returncode = process.poll(); timed_out = returncode is None and time.monotonic() - job["started"] > timeout_seconds
            if returncode is None and not timed_out: continue
            if timed_out: process.kill(); returncode = process.wait(timeout=15)
            stdout, stderr = process.communicate(); active.pop(process)
            event: dict[str, Any] = {"sample_id": job["sample_id"], "attempt": job["attempt"], "returncode": int(returncode), "duration_seconds": round(time.monotonic() - job["started"], 3), "timed_out": timed_out, "access_violation": int(returncode) in ACCESS_VIOLATION_CODES, "stdout": stdout[-2048:], "stderr": stderr[-8192:]}
            try:
                if returncode != 0: raise RuntimeError(f"validator exited {returncode}")
                result = json.loads(stdout)
                if result.get("passed") is not True or result.get("validated_record_count") != 1 or result.get("semantic_sha256") != manifest["semantic_sha256"] or bool(result.get("replay")) is not replay:
                    raise ValueError("isolated validator result authority drifted")
                event["passed"] = True
            except Exception as error:
                event["passed"] = False; event["failure"] = f"{type(error).__name__}: {error}"
                if job["attempt"] >= max_attempts:
                    raise RuntimeError(f"Neural motion validator for {job['sample_id']} exhausted {max_attempts} attempts") from error
                jobs.append(job)
            events.append(event)
    if sum(bool(event.get("passed")) for event in events) != len(sample_ids):
        raise RuntimeError("Neural motion isolated validator coverage drifted.")
    return events


def validate_corpus_resilient(output: Path = DEFAULT_CORPUS, *, replay: bool = False, workers: int = 2, max_attempts: int = 3, timeout_seconds: int = 600) -> dict[str, Any]:
    output = Path(output).resolve()
    if type(workers) is not int or not 1 <= workers <= 2 or type(max_attempts) is not int or not 1 <= max_attempts <= 3 or type(timeout_seconds) is not int or not 30 <= timeout_seconds <= 1800:
        raise ValueError("Neural motion resilient validation policy drifted.")
    manifest = _read_canonical_json(output / "neural_cell_motion_corpus.json"); sample_ids = [record["sample_id"] for record in manifest["records"]]
    events = _validate_records_isolated(output, sample_ids, replay=replay, workers=workers, max_attempts=max_attempts, timeout_seconds=timeout_seconds)
    report: dict[str, Any] = {"format": VALIDATION_FORMAT, "status": "passed", "source_sha256": corpus_source_sha256(), "corpus_semantic_sha256": manifest["semantic_sha256"], "replay": replay, "identity_count": len(sample_ids), "validated_identity_count": len(sample_ids), "attempt_count": len(events), "retry_count": len(events) - len(sample_ids), "native_failure_count": sum(bool(event["access_violation"]) for event in events), "timeout_count": sum(bool(event["timed_out"]) for event in events), "events": events}
    report["semantic_sha256"] = sha256_bytes(canonical_json_bytes(report)); report_path = output.parent / f"{output.name}_validation_telemetry.json"; _atomic_bytes(report_path, canonical_json_bytes(report))
    return {"passed": True, "replay": replay, "identity_count": len(sample_ids), "sample_count": manifest["scope"]["sample_count"], "semantic_sha256": manifest["semantic_sha256"], "validation_telemetry_sha256": sha256_file(report_path), "retry_count": report["retry_count"]}


def build_corpus_resilient(
    output: Path = DEFAULT_CORPUS, *, identities_per_family: int | None = None,
    workers: int = 2, max_attempts: int = 3, timeout_seconds: int = 600,
    recover_from: list[Path] | None = None,
) -> dict[str, Any]:
    output = Path(output).resolve(); recover_roots = [Path(path).resolve() for path in (recover_from or [])]
    if output.exists(): raise FileExistsError(output)
    if type(workers) is not int or not 1 <= workers <= 2 or type(max_attempts) is not int or not 1 <= max_attempts <= 3:
        raise ValueError("Neural motion resilient worker policy drifted.")
    if type(timeout_seconds) is not int or not 30 <= timeout_seconds <= 1800:
        raise ValueError("Neural motion worker timeout drifted.")
    output.parent.mkdir(parents=True, exist_ok=True)
    require_disk_floor(output.parent, floor_gb=100, planned_bytes=2 * 1024**3)
    build_root = (PROJECT_ROOT / "work" / f"neural-cell-motion-build-{uuid.uuid4().hex}").resolve(); build_root.mkdir(parents=True, exist_ok=False)
    preflight_events: list[dict[str, Any]] = []
    for preflight_attempt in range(1, max_attempts + 1):
        destination = build_root / "preflight" / f"attempt_{preflight_attempt:02d}"
        started = time.monotonic()
        event: dict[str, Any] = {"attempt": preflight_attempt, "returncode": -1, "duration_seconds": 0.0, "timed_out": False, "access_violation": False, "stdout": "", "stderr": ""}
        try:
            preflight = subprocess.run([sys.executable, "-m", "forge.neural_cell_motion.worker", "--preflight", "--destination", str(destination)], cwd=PROJECT_ROOT, env=_worker_environment(), capture_output=True, text=True, timeout=min(timeout_seconds, 300), creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            event = {"attempt": preflight_attempt, "returncode": preflight.returncode, "duration_seconds": round(time.monotonic() - started, 3), "timed_out": False, "access_violation": preflight.returncode in ACCESS_VIOLATION_CODES, "stdout": preflight.stdout[-2048:], "stderr": preflight.stderr[-8192:]}
            if preflight.returncode != 0: raise RuntimeError(f"preflight exited {preflight.returncode}")
            result = _read_canonical_json(destination / "preflight.json", maximum_bytes=128 * 1024)
            if result["format"] != PREFLIGHT_FORMAT or result["status"] != "passed" or result["source_sha256"] != corpus_source_sha256() or result["semantic_sha256"] != sha256_bytes(canonical_json_bytes({key: value for key, value in result.items() if key != "semantic_sha256"})):
                raise ValueError("Neural motion preflight authority drifted.")
            event["passed"] = True; preflight_events.append(event); break
        except subprocess.TimeoutExpired as error:
            raw_stdout, raw_stderr = error.stdout or "", error.stderr or ""
            if isinstance(raw_stdout, bytes): raw_stdout = raw_stdout.decode("utf-8", errors="replace")
            if isinstance(raw_stderr, bytes): raw_stderr = raw_stderr.decode("utf-8", errors="replace")
            event.update({"duration_seconds": round(time.monotonic() - started, 3), "timed_out": True, "stdout": raw_stdout[-2048:], "stderr": raw_stderr[-8192:], "passed": False, "failure": "TimeoutExpired: isolated preflight exceeded its deadline"}); preflight_events.append(event)
        except Exception as error:
            event["duration_seconds"] = round(time.monotonic() - started, 3); event["passed"] = False; event["failure"] = f"{type(error).__name__}: {error}"; preflight_events.append(event)
    else:
        raise RuntimeError(f"Neural motion isolated preflight exhausted {max_attempts} attempts; evidence preserved at {build_root}")
    motion_path, anatomy_path = DEFAULT_MOTION.resolve(), DEFAULT_ANATOMY.resolve()
    motion = _read_canonical_json(motion_path, maximum_bytes=64 * 1024 * 1024); anatomy = _read_canonical_json(anatomy_path, maximum_bytes=64 * 1024 * 1024)
    selected, totals, production = _selection_plan(anatomy["offspring"], identities_per_family)
    corpus_stage = build_root / "corpus"; shards = corpus_stage / "shards"; specs = build_root / "specs"; attempts_root = build_root / "attempts"
    for path in (shards, specs, attempts_root): path.mkdir(parents=True, exist_ok=False)
    motion_sha, anatomy_sha = sha256_file(motion_path), sha256_file(anatomy_path); jobs: list[dict[str, Any]] = []
    for record, ordinal in selected:
        family_id = int(record["family_id"]); split = ("test" if ordinal == totals[family_id] - 1 else "validation" if ordinal == totals[family_id] - 2 else "train") if production else "smoke"
        spec_path = specs / f"{record['sample_id']}.json"; spec = _write_spec(spec_path, record, ordinal, split, identities_per_family, motion_sha, anatomy_sha)
        reuse = next((root / "shards" / f"{record['sample_id']}.npz" for root in recover_roots if (root / "shards" / f"{record['sample_id']}.npz").is_file()), None)
        jobs.append({"record": record, "spec": spec, "spec_path": spec_path, "reuse": reuse, "attempt": 0})
    active: dict[subprocess.Popen[str], dict[str, Any]] = {}; completed: dict[str, dict[str, Any]] = {}; telemetry: list[dict[str, Any]] = []
    while jobs or active:
        while jobs and len(active) < workers:
            job = jobs.pop(0); job["attempt"] += 1; sample_id = job["record"]["sample_id"]; attempt_dir = attempts_root / sample_id / f"attempt_{job['attempt']:02d}"; attempt_dir.mkdir(parents=True)
            command = [sys.executable, "-m", "forge.neural_cell_motion.worker", "--spec", str(job["spec_path"]), "--destination", str(attempt_dir)]
            if job["reuse"] is not None: command.extend(["--reuse", str(job["reuse"])])
            process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=_worker_environment(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            job.update({"attempt_dir": attempt_dir, "started": time.monotonic()}); active[process] = job
        if not active: break
        time.sleep(.1)
        for process, job in list(active.items()):
            returncode = process.poll()
            timed_out = returncode is None and time.monotonic() - job["started"] > timeout_seconds
            if returncode is None and not timed_out: continue
            if timed_out:
                process.kill(); returncode = process.wait(timeout=15)
            stdout, stderr = process.communicate(); active.pop(process); sample_id = job["record"]["sample_id"]
            event = {"sample_id": sample_id, "attempt": job["attempt"], "returncode": int(returncode), "duration_seconds": round(time.monotonic() - job["started"], 3), "timed_out": timed_out, "access_violation": int(returncode) in ACCESS_VIOLATION_CODES, "recovery_requested": job["reuse"] is not None, "stdout": stdout[-2048:], "stderr": stderr[-8192:]}
            telemetry.append(event)
            try:
                if returncode != 0: raise RuntimeError(f"worker exited {returncode}")
                result = _read_canonical_json(job["attempt_dir"] / "result.json", maximum_bytes=256 * 1024)
                if result["format"] != RESULT_FORMAT or result["status"] != "passed" or result["source_sha256"] != corpus_source_sha256() or result["spec_sha256"] != job["spec"]["semantic_sha256"] or result["semantic_sha256"] != sha256_bytes(canonical_json_bytes({key: value for key, value in result.items() if key != "semantic_sha256"})):
                    raise ValueError("worker result authority drifted")
                artifact = job["attempt_dir"] / f"{sample_id}.npz"; destination = shards / artifact.name; os.replace(artifact, destination); completed[sample_id] = result["record"]; event["passed"] = True; event["reused"] = bool(result["reused"])
            except Exception as error:
                event["passed"] = False; event["failure"] = f"{type(error).__name__}: {error}"
                if job["attempt"] >= max_attempts:
                    raise RuntimeError(f"Neural motion shard {sample_id} exhausted {max_attempts} attempts; evidence preserved at {build_root}") from error
                # A source-bound recovery shard that triggers a native worker
                # crash remains useful and is retried in a fresh process. Only
                # an explicit exact-replay rejection proves that the shard
                # itself must be discarded and regenerated.
                if "recovery shard failed exact replay" in stderr or returncode == 0:
                    job["reuse"] = None
                jobs.append(job)
    ordered_records = [completed[record["sample_id"]] for record, _ in selected]
    if len(ordered_records) != len(selected): raise RuntimeError(f"Neural motion resilient build incomplete; evidence preserved at {build_root}")
    _write_corpus_manifest(corpus_stage, motion_path=motion_path, anatomy_path=anatomy_path, motion=motion, anatomy=anatomy, identities_per_family=identities_per_family, records=ordered_records)
    validation_events = _validate_records_isolated(corpus_stage, [record["sample_id"] for record in ordered_records], replay=False, workers=workers, max_attempts=max_attempts, timeout_seconds=min(timeout_seconds, 300))
    # Windows can reject a directory rename across two watched parent
    # directories even on one volume. Publish through a hidden sibling so the
    # only operation that makes the corpus visible is a same-parent rename.
    publication_stage = output.parent / f".{output.name}.publish-{uuid.uuid4().hex}"
    shutil.copytree(corpus_stage, publication_stage)
    publication_events = _validate_records_isolated(publication_stage, [record["sample_id"] for record in ordered_records], replay=False, workers=workers, max_attempts=max_attempts, timeout_seconds=min(timeout_seconds, 300))
    for event in publication_events: event["publication_stage"] = True
    validation_events.extend(publication_events)
    for publication_attempt in range(1, max_attempts + 1):
        try:
            os.replace(publication_stage, output); break
        except PermissionError:
            if publication_attempt >= max_attempts: raise
            time.sleep(.25 * publication_attempt)
    output_label = output.relative_to(PROJECT_ROOT).as_posix() if output.is_relative_to(PROJECT_ROOT) else str(output)
    report: dict[str, Any] = {"format": SUPERVISOR_FORMAT, "status": "passed", "source_sha256": corpus_source_sha256(), "output": output_label, "identity_count": len(ordered_records), "sample_count": len(ordered_records) * 944, "workers": workers, "max_attempts": max_attempts, "timeout_seconds": timeout_seconds, "preflight_events": preflight_events, "attempt_count": len(telemetry), "retry_count": len(telemetry) - len(ordered_records), "native_failure_count": sum(bool(item["access_violation"]) for item in telemetry) + sum(bool(item["access_violation"]) for item in preflight_events), "timeout_count": sum(bool(item["timed_out"]) for item in telemetry) + sum(bool(item["timed_out"]) for item in preflight_events), "recovered_count": sum(bool(item.get("reused")) for item in telemetry if item.get("passed")), "events": telemetry, "validation_events": validation_events}
    report["semantic_sha256"] = hashlib.sha256(canonical_json_bytes(report)).hexdigest(); report_path = output.parent / f"{output.name}_build_telemetry.json"; _atomic_bytes(report_path, canonical_json_bytes(report))
    return {"passed": True, "identity_count": len(ordered_records), "sample_count": len(ordered_records) * 944, "semantic_sha256": _read_canonical_json(output / "neural_cell_motion_corpus.json")["semantic_sha256"], "build_telemetry_sha256": sha256_file(report_path), "recovered_count": report["recovered_count"], "retry_count": report["retry_count"]}
