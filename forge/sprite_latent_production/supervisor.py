from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

from .checkpoint import load_checkpoint
from .contract import (
    CALIBRATION_FORMAT,
    DEFAULT_CORPUS,
    DEFAULT_OUTPUT,
    FORMAT,
    MIN_FREE_BYTES,
    ProductionConfig,
    SEGMENT_FORMAT,
    canonical_json_bytes,
    production_source_hash,
    sha256_bytes,
    sha256_file,
)


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update({
        "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
        "NUMEXPR_NUM_THREADS": "2",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    })
    return environment


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _load_report(
    path: Path,
    *,
    expected_config: ProductionConfig | None = None,
    expected_start_epoch: int | None = None,
    expected_requested_end_epoch: int | None = None,
    expected_previous_sha256: str | None = None,
    allow_partial: bool = False,
) -> dict[str, Any]:
    raw = Path(path).read_bytes(); report = json.loads(raw)
    if raw != canonical_json_bytes(report): raise ValueError("segment report is not canonical JSON")
    unsigned = dict(report); stored = unsigned.pop("report_sha256", None)
    if stored != sha256_bytes(canonical_json_bytes(unsigned)): raise ValueError("segment report self-hash mismatch")
    if report.get("format") != SEGMENT_FORMAT or report.get("status") != "passed" or report.get("source_sha256") != production_source_hash():
        raise ValueError("segment report authority mismatch")
    config = ProductionConfig.from_metadata(dict(report["config"]))
    if expected_config is not None and config != expected_config:
        raise ValueError("segment report config mismatch")
    if expected_start_epoch is not None and int(report["start_epoch"]) != expected_start_epoch:
        raise ValueError("segment report start epoch mismatch")
    if expected_requested_end_epoch is not None and int(report["requested_end_epoch"]) != expected_requested_end_epoch:
        raise ValueError("segment report requested end epoch mismatch")
    if bool(report["stopped_early"]) and not allow_partial:
        raise ValueError("partial segment cannot enter the production checkpoint chain")
    if not isinstance(report.get("gates"), dict) or not all(report["gates"].values()):
        raise ValueError("segment report hard gate failed")
    checkpoint = Path(path).parent / report["checkpoint"]["path"]
    if checkpoint.stat().st_size != report["checkpoint"]["bytes"] or sha256_file(checkpoint) != report["checkpoint"]["sha256"]: raise ValueError("segment checkpoint artifact mismatch")
    loaded = load_checkpoint(checkpoint)
    if (
        loaded["ema_state_sha256"] != report["checkpoint"]["ema_state_sha256"]
        or loaded["model_state_sha256"] != report["checkpoint"]["model_state_sha256"]
        or loaded["config"] != report["config"]
        or int(loaded["epoch"]) != int(report["end_epoch"])
        or int(loaded["global_step"]) != int(report["global_step"])
        or len(loaded["history"]) != int(report["checkpoint_history_length"])
    ):
        raise ValueError("segment checkpoint semantic mismatch")
    if loaded["previous_checkpoint_sha256"] != expected_previous_sha256:
        raise ValueError("segment checkpoint predecessor mismatch")
    if bool(report["stopped_early"]) != (loaded["partial_epoch"] is not None or int(report["end_epoch"]) < int(report["requested_end_epoch"])):
        raise ValueError("segment partial-progress contract mismatch")
    if report["corpus_sha256"] != loaded["corpus_sha256"] or report["split_fingerprint"] != loaded["split_fingerprint"] or report["legal_tuple_fingerprint"] != loaded["legal_tuple_fingerprint"]:
        raise ValueError("segment report frozen provenance mismatch")
    evaluation = report.get("evaluation")
    if not isinstance(evaluation, dict) or evaluation.get("legal_projection_fraction") != 1.0:
        raise ValueError("segment evaluation safety contract failed")
    return report


def _launch_worker(
    *, root: Path, corpus: Path, config: ProductionConfig, start_epoch: int,
    end_epoch: int, resume: Path | None, label: str, max_steps: int | None = None,
) -> tuple[Path, list[dict[str, Any]]]:
    staging = root / "staging"; logs = root / "logs"; staging.mkdir(parents=True, exist_ok=True); logs.mkdir(parents=True, exist_ok=True)
    events = []
    for attempt in range(1, config.max_attempts + 1):
        destination = staging / f"{label}_attempt_{attempt:02d}"
        log_path = logs / f"{label}_attempt_{attempt:02d}.log"
        if destination.exists() or log_path.exists():
            continue
        command = [
            sys.executable, "-m", "forge.sprite_latent_production", "worker",
            "--corpus", str(corpus), "--output", str(destination),
            "--start-epoch", str(start_epoch), "--end-epoch", str(end_epoch),
            "--config-json", json.dumps(config.metadata(), separators=(",", ":")),
        ]
        if resume is not None: command.extend(("--resume", str(resume)))
        if max_steps is not None: command.extend(("--max-steps", str(max_steps)))
        started = time.monotonic()
        with log_path.open("xb") as log:
            process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[2], env=_environment(), stdout=log, stderr=subprocess.STDOUT)
            try:
                return_code = process.wait(timeout=config.worker_timeout_seconds)
                timed_out = False
            except subprocess.TimeoutExpired:
                process.kill(); return_code = process.wait(timeout=30); timed_out = True
        event = {
            "label": label, "attempt": attempt, "return_code": int(return_code),
            "elapsed_seconds": round(time.monotonic() - started, 3), "timed_out": timed_out,
            "access_violation": int(return_code) in (3221225477, -1073741819),
            "log": log_path.relative_to(root).as_posix(),
        }
        try:
            if return_code != 0 or timed_out: raise RuntimeError("worker process failed")
            _load_report(
                destination / "segment_report.json",
                expected_config=config,
                expected_start_epoch=start_epoch,
                expected_requested_end_epoch=end_epoch,
                expected_previous_sha256=sha256_file(resume) if resume is not None else None,
                allow_partial=max_steps is not None,
            )
            event["status"] = "accepted"; events.append(event); return destination, events
        except (OSError, ValueError, RuntimeError) as error:
            event["status"] = "rejected"; event["error"] = str(error)[:1000]; events.append(event)
    raise RuntimeError(f"sprite latent worker {label} exhausted retries: {events}")


def run_calibration(
    output: Path = DEFAULT_OUTPUT,
    *, corpus: Path = DEFAULT_CORPUS,
    config: ProductionConfig = ProductionConfig(),
    steps: int = 100,
) -> dict[str, Any]:
    root = Path(output).resolve(); root.mkdir(parents=True, exist_ok=True)
    final = root / "calibration"
    if final.exists():
        report = _load_report(
            final / "segment_report.json",
            expected_config=config,
            expected_start_epoch=0,
            expected_requested_end_epoch=config.segment_epochs,
            expected_previous_sha256=None,
            allow_partial=True,
        )
        if int(report["training_steps"]) != steps or not bool(report["stopped_early"]):
            raise ValueError("existing calibration does not match the requested bounded run")
        return {"status": "reused", "report": report, "events": []}
    staged, events = _launch_worker(root=root, corpus=Path(corpus).resolve(), config=config, start_epoch=0, end_epoch=config.segment_epochs, resume=None, label="calibration", max_steps=steps)
    os.replace(staged, final)
    report = _load_report(
        final / "segment_report.json",
        expected_config=config,
        expected_start_epoch=0,
        expected_requested_end_epoch=config.segment_epochs,
        expected_previous_sha256=None,
        allow_partial=True,
    )
    summary = {
        "format": CALIBRATION_FORMAT, "status": "passed", "source_sha256": production_source_hash(),
        "steps": steps, "worker_report_sha256": report["report_sha256"], "events": events,
        "throughput": report["steps_per_second"], "peak_reserved_bytes": report["peak_reserved_bytes"],
        "evaluation": report["evaluation"],
    }
    summary["calibration_sha256"] = sha256_bytes(canonical_json_bytes(summary))
    _write_atomic(root / "calibration_report.json", canonical_json_bytes(summary))
    return {"status": "passed", "report": report, "events": events, "summary": summary}


def run_supervisor(
    output: Path = DEFAULT_OUTPUT,
    *, corpus: Path = DEFAULT_CORPUS,
    config: ProductionConfig = ProductionConfig(),
) -> dict[str, Any]:
    root = Path(output).resolve(); root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root).free < MIN_FREE_BYTES + 1024**3: raise OSError("production training would approach the disk floor")
    final_manifest = root / "production_manifest.json"
    if final_manifest.exists():
        return validate_production_manifest(final_manifest)
    calibration = run_calibration(root, corpus=corpus, config=config)
    segments_root = root / "segments"; segments_root.mkdir(exist_ok=True)
    all_events = list(calibration.get("events", [])); segment_reports = []
    resume: Path | None = None; expected_start = 0
    for end_epoch in range(config.segment_epochs, config.epochs + 1, config.segment_epochs):
        if shutil.disk_usage(root).free < MIN_FREE_BYTES + 1024**3:
            raise OSError("production training would approach the disk floor")
        target = segments_root / f"epoch_{end_epoch:03d}"
        previous_sha256 = sha256_file(resume) if resume is not None else None
        if target.exists():
            report = _load_report(
                target / "segment_report.json",
                expected_config=config,
                expected_start_epoch=expected_start,
                expected_requested_end_epoch=end_epoch,
                expected_previous_sha256=previous_sha256,
            )
            if int(report["end_epoch"]) != end_epoch: raise ValueError("existing segment chain is not contiguous")
        else:
            staged, events = _launch_worker(root=root, corpus=Path(corpus).resolve(), config=config, start_epoch=expected_start, end_epoch=end_epoch, resume=resume, label=f"epoch_{end_epoch:03d}")
            all_events.extend(events); os.replace(staged, target)
            report = _load_report(
                target / "segment_report.json",
                expected_config=config,
                expected_start_epoch=expected_start,
                expected_requested_end_epoch=end_epoch,
                expected_previous_sha256=previous_sha256,
            )
        resume = target / "checkpoint.pt"; expected_start = end_epoch; segment_reports.append(report)
        pointer = {"format": "nullvector-sprite-fsq-latest-pointer-v1", "epoch": end_epoch, "checkpoint": str(resume.relative_to(root)).replace("\\", "/"), "sha256": sha256_file(resume), "source_sha256": production_source_hash()}
        _write_atomic(root / "latest.json", canonical_json_bytes(pointer))
    accepted = [report for report in segment_reports if report["evaluation"]["quality_accepted"]]
    best = max(accepted or segment_reports, key=lambda report: (report["evaluation"]["quality_score"], -report["end_epoch"]))
    manifest = {
        "format": FORMAT, "status": "ready" if accepted else "quality_failed", "source_sha256": production_source_hash(),
        "config": config.metadata(), "corpus_sha256": best["corpus_sha256"], "split_fingerprint": best["split_fingerprint"], "legal_tuple_fingerprint": best["legal_tuple_fingerprint"],
        "counts": {"segments": len(segment_reports), "epochs": config.epochs, "global_steps": segment_reports[-1]["global_step"], "accepted_checkpoints": len(accepted)},
        "segments": [{"start_epoch": report["start_epoch"], "end_epoch": report["end_epoch"], "report_sha256": report["report_sha256"], "checkpoint_sha256": report["checkpoint"]["sha256"], "quality_score": report["evaluation"]["quality_score"], "quality_accepted": report["evaluation"]["quality_accepted"]} for report in segment_reports],
        "best": {"epoch": best["end_epoch"], "checkpoint": f"segments/epoch_{int(best['end_epoch']):03d}/checkpoint.pt", "checkpoint_sha256": best["checkpoint"]["sha256"], "quality_score": best["evaluation"]["quality_score"], "evaluation": best["evaluation"]},
        "latest": {"epoch": segment_reports[-1]["end_epoch"], "checkpoint": f"segments/epoch_{config.epochs:03d}/checkpoint.pt", "checkpoint_sha256": segment_reports[-1]["checkpoint"]["sha256"], "evaluation": segment_reports[-1]["evaluation"]},
        "calibration": {"report_sha256": calibration["report"]["report_sha256"], "steps_per_second": calibration["report"]["steps_per_second"], "peak_reserved_bytes": calibration["report"]["peak_reserved_bytes"]},
        "telemetry": {"attempts": len(all_events), "retries": sum(event["attempt"] > 1 for event in all_events), "access_violations": sum(bool(event["access_violation"]) for event in all_events), "events": all_events},
        "gates": {"all_segments_complete": len(segment_reports) == config.epochs // config.segment_epochs, "checkpoint_chain_contiguous": expected_start == config.epochs, "legal_projection_exact_every_segment": all(report["evaluation"]["legal_projection_fraction"] == 1.0 for report in segment_reports), "full_quality_accepted": bool(accepted)},
    }
    manifest["manifest_sha256"] = sha256_bytes(canonical_json_bytes(manifest))
    _write_atomic(final_manifest, canonical_json_bytes(manifest))
    return validate_production_manifest(final_manifest)


def validate_production_manifest(path: Path) -> dict[str, Any]:
    manifest_path = Path(path).resolve()
    raw = manifest_path.read_bytes(); manifest = json.loads(raw)
    if raw != canonical_json_bytes(manifest):
        raise ValueError("production manifest is not canonical JSON")
    unsigned = dict(manifest); stored = unsigned.pop("manifest_sha256", None)
    if stored != sha256_bytes(canonical_json_bytes(unsigned)):
        raise ValueError("production manifest self-hash mismatch")
    if manifest.get("format") != FORMAT or manifest.get("source_sha256") != production_source_hash():
        raise ValueError("production manifest authority mismatch")
    config = ProductionConfig.from_metadata(dict(manifest["config"]))
    root = manifest_path.parent
    reports: list[dict[str, Any]] = []
    previous_sha256: str | None = None
    expected_start = 0
    for entry in manifest["segments"]:
        end_epoch = int(entry["end_epoch"])
        report_path = root / "segments" / f"epoch_{end_epoch:03d}" / "segment_report.json"
        report = _load_report(
            report_path,
            expected_config=config,
            expected_start_epoch=expected_start,
            expected_requested_end_epoch=end_epoch,
            expected_previous_sha256=previous_sha256,
        )
        if (
            report["report_sha256"] != entry["report_sha256"]
            or report["checkpoint"]["sha256"] != entry["checkpoint_sha256"]
            or bool(report["evaluation"]["quality_accepted"]) != bool(entry["quality_accepted"])
        ):
            raise ValueError("production manifest segment summary mismatch")
        reports.append(report)
        previous_sha256 = report["checkpoint"]["sha256"]
        expected_start = end_epoch
    if len(reports) != config.epochs // config.segment_epochs or expected_start != config.epochs:
        raise ValueError("production manifest segment coverage mismatch")
    best_epoch = int(manifest["best"]["epoch"])
    best = next((report for report in reports if int(report["end_epoch"]) == best_epoch), None)
    if best is None or best["checkpoint"]["sha256"] != manifest["best"]["checkpoint_sha256"] or best["evaluation"] != manifest["best"]["evaluation"]:
        raise ValueError("production manifest best checkpoint mismatch")
    latest = reports[-1]
    if latest["checkpoint"]["sha256"] != manifest["latest"]["checkpoint_sha256"] or latest["evaluation"] != manifest["latest"]["evaluation"]:
        raise ValueError("production manifest latest checkpoint mismatch")
    accepted = sum(bool(report["evaluation"]["quality_accepted"]) for report in reports)
    expected_status = "ready" if accepted else "quality_failed"
    if manifest["status"] != expected_status or int(manifest["counts"]["accepted_checkpoints"]) != accepted:
        raise ValueError("production manifest quality verdict mismatch")
    if not all(bool(value) for key, value in manifest["gates"].items() if key != "full_quality_accepted") or bool(manifest["gates"]["full_quality_accepted"]) != bool(accepted):
        raise ValueError("production manifest hard gate mismatch")
    return manifest
