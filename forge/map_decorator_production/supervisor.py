from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from ..config import PROJECT_ROOT
from ..map_decorator_ml.checkpoint import file_sha256
from ..safety import require_disk_floor
from .contract import DISK_FLOOR_GIB, MAX_PROCESS_ATTEMPTS, TRAINING_FORMAT_VERSION
from .training import ProductionTrainingConfig


def _atomic_json(path: Path, payload: object) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    require_disk_floor(path.parent, floor_gb=DISK_FLOOR_GIB, planned_bytes=len(encoded) + 1024 * 1024)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
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


def _run_attempts(
    command: list[str],
    *,
    label: str,
    output_root: Path,
    max_attempts: int = MAX_PROCESS_ATTEMPTS,
) -> list[dict[str, object]]:
    attempts: list[dict[str, object]] = []
    logs = output_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, max_attempts + 1):
        require_disk_floor(output_root, floor_gb=DISK_FLOOR_GIB, planned_bytes=3 * 1024**3)
        stdout = logs / f"{label}-attempt{attempt:02d}.stdout.log"
        stderr = logs / f"{label}-attempt{attempt:02d}.stderr.log"
        started = time.perf_counter()
        with stdout.open("xb") as stdout_handle, stderr.open("xb") as stderr_handle:
            process = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                stdin=subprocess.DEVNULL,
                stdout=stdout_handle,
                stderr=stderr_handle,
                check=False,
            )
        record = {
            "label": label,
            "attempt": attempt,
            "returncode": process.returncode,
            "returncode_unsigned_hex": f"0x{process.returncode & 0xFFFF_FFFF:08x}",
            "native_failure": _native_label(process.returncode),
            "elapsed_seconds": time.perf_counter() - started,
            "stdout": stdout.relative_to(output_root).as_posix(),
            "stdout_sha256": file_sha256(stdout),
            "stderr": stderr.relative_to(output_root).as_posix(),
            "stderr_sha256": file_sha256(stderr),
            "passed": process.returncode == 0,
        }
        attempts.append(record)
        if process.returncode == 0:
            return attempts
    raise RuntimeError(
        f"Training phase {label} failed after {max_attempts} attempts; "
        f"last exit={attempts[-1]['returncode_unsigned_hex']}."
    )


def _segment_report_path(root: Path, start: int, stop: int) -> Path:
    return root / "segments" / f"epochs-{start + 1:03d}-{stop:03d}" / "segment_report.json"


def _checkpoint_path(root: Path, start: int, stop: int) -> Path:
    return _segment_report_path(root, start, stop).parent / f"checkpoint_epoch_{stop:03d}.pt"


def run_supervisor(
    corpus: Path,
    output: Path,
    *,
    python: Path = Path(sys.executable),
    config: ProductionTrainingConfig = ProductionTrainingConfig(),
    run_calibration: bool = True,
) -> dict[str, object]:
    corpus = Path(corpus).resolve()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    final_report = output / "training_supervisor_report.json"
    if final_report.exists():
        raise FileExistsError("Production training supervisor report already exists.")
    telemetry: list[dict[str, object]] = []
    calibration_path = output / "calibration_100step_bf16.json"
    if run_calibration and not calibration_path.exists():
        command = [
            str(python),
            "-m",
            "forge.map_decorator_production.training",
            "calibrate",
            "--corpus",
            str(corpus),
            "--output",
            str(calibration_path),
            "--steps",
            "100",
            "--batch-size",
            str(config.batch_size),
        ]
        telemetry.extend(
            _run_attempts(command, label="calibration-100step-bf16", output_root=output)
        )
    if run_calibration:
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        if not calibration.get("passed") or not calibration.get("loss", {}).get("finite"):
            raise RuntimeError("CUDA BF16 calibration did not pass its finite-loss gate.")
    else:
        calibration = None

    segment_reports: list[dict[str, object]] = []
    resume: Path | None = None
    previous_checkpoint_hash: str | None = None
    for start in range(0, config.epochs, config.segment_epochs):
        stop = start + config.segment_epochs
        report_path = _segment_report_path(output, start, stop)
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
        else:
            command = [
                str(python),
                "-m",
                "forge.map_decorator_production.training",
                "segment",
                "--corpus",
                str(corpus),
                "--output-root",
                str(output),
                "--start-epoch",
                str(start),
                "--stop-epoch",
                str(stop),
                "--epochs",
                str(config.epochs),
                "--batch-size",
                str(config.batch_size),
                "--train-steps-per-epoch",
                str(config.train_steps_per_epoch),
            ]
            if resume is not None:
                command.extend(("--resume", str(resume)))
            telemetry.extend(
                _run_attempts(
                    command,
                    label=f"segment-epochs-{start + 1:03d}-{stop:03d}",
                    output_root=output,
                )
            )
            report = json.loads(report_path.read_text(encoding="utf-8"))
        checkpoint = _checkpoint_path(output, start, stop)
        if not report.get("passed") or not checkpoint.is_file():
            raise RuntimeError(f"Segment {start + 1}-{stop} did not publish a valid report/checkpoint.")
        checkpoint_hash = file_sha256(checkpoint)
        if checkpoint_hash != report["checkpoint"]["checkpoint_sha256"]:
            raise RuntimeError("Segment checkpoint hash disagrees with its immutable report.")
        if start and report.get("resume_checkpoint") != str(resume.resolve()):  # type: ignore[union-attr]
            raise RuntimeError("Segment chain does not name the exact prior immutable checkpoint.")
        if report["held_out"]["hard_legality"] != 1.0 or report["sentinel"]["hard_legality"] != 1.0:
            raise RuntimeError("Segment hard held-out/sentinel gate failed.")
        report["checkpoint_file_sha256_verified"] = checkpoint_hash
        report["prior_checkpoint_sha256"] = previous_checkpoint_hash
        segment_reports.append(report)
        previous_checkpoint_hash = checkpoint_hash
        resume = checkpoint

    summary: dict[str, object] = {
        "format_version": TRAINING_FORMAT_VERSION,
        "passed": True,
        "corpus": str(corpus),
        "output": str(output),
        "config": config.to_dict(),
        "calibration": calibration,
        "segment_count": len(segment_reports),
        "segments": segment_reports,
        "final_checkpoint": str(resume),
        "final_checkpoint_sha256": previous_checkpoint_hash,
        "telemetry": telemetry,
        "native_failure_count": sum(bool(item["native_failure"]) for item in telemetry),
        "exact_two_epoch_segments": all(
            report["stop_epoch"] - report["start_epoch"] == 2 for report in segment_reports
        ),
        "all_held_out_safety_gates": all(
            report["held_out"]["hard_legality"] == 1.0 for report in segment_reports
        ),
        "all_sentinel_safety_gates": all(
            report["sentinel"]["hard_legality"] == 1.0 for report in segment_reports
        ),
    }
    _atomic_json(final_report, summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fresh-process two-epoch CUDA training supervisor")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--train-steps-per-epoch", type=int, default=256)
    parser.add_argument("--skip-calibration", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = ProductionTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        train_steps_per_epoch=args.train_steps_per_epoch,
    )
    report = run_supervisor(
        args.corpus,
        args.output,
        python=args.python,
        config=config,
        run_calibration=not args.skip_calibration,
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

