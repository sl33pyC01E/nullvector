from __future__ import annotations

import argparse
import json
from pathlib import Path

from .smoke import run_cpu_smoke, validate_cpu_smoke
from .pilot import RealCorpusPilotConfig, run_real_corpus_pilot, validate_real_corpus_pilot
from .calibration import (
    CalibrationConfig,
    run_calibration_worker,
    supervise_calibration,
    validate_calibration,
    validate_supervised_calibration,
)
from .contract import LocatorModelConfig


def main() -> int:
    parser = argparse.ArgumentParser(description="Map decorator v3 sparse-localization foundation")
    sub = parser.add_subparsers(dest="command", required=True)
    smoke = sub.add_parser("smoke"); smoke.add_argument("--output", type=Path, required=True); smoke.add_argument("--steps", type=int, default=4)
    validate = sub.add_parser("validate"); validate.add_argument("report", type=Path); validate.add_argument("--exact-replay", action="store_true")
    pilot = sub.add_parser("pilot")
    pilot.add_argument("--corpus", type=Path, required=True)
    pilot.add_argument("--index", type=Path, required=True)
    pilot.add_argument("--output", type=Path, required=True)
    pilot.add_argument("--steps", type=int, default=4)
    pilot.add_argument("--eval-samples", type=int, default=4)
    pilot_validate = sub.add_parser("validate-pilot")
    pilot_validate.add_argument("report", type=Path)
    pilot_validate.add_argument("--corpus", type=Path, required=True)
    pilot_validate.add_argument("--index", type=Path, required=True)
    pilot_validate.add_argument("--exact-replay", action="store_true")
    worker = sub.add_parser("calibration-worker")
    worker.add_argument("--corpus", type=Path, required=True)
    worker.add_argument("--index", type=Path, required=True)
    worker.add_argument("--output", type=Path, required=True)
    worker.add_argument("--steps", type=int, default=100)
    worker.add_argument("--validation-batch-size", type=int, default=4)
    worker.add_argument("--test-batch-size", type=int, default=4)
    worker.add_argument("--base-channels", type=int, default=48)
    calibrate = sub.add_parser("calibrate")
    calibrate.add_argument("--corpus", type=Path, required=True)
    calibrate.add_argument("--index", type=Path, required=True)
    calibrate.add_argument("--output", type=Path, required=True)
    calibrate.add_argument("--steps", type=int, default=100)
    calibrate.add_argument("--validation-batch-size", type=int, default=4)
    calibrate.add_argument("--test-batch-size", type=int, default=4)
    calibrate.add_argument("--base-channels", type=int, default=48)
    calibrate.add_argument("--max-attempts", type=int, default=3)
    calibrate.add_argument("--timeout-seconds", type=int, default=3600)
    calibration_validate = sub.add_parser("validate-calibration")
    calibration_validate.add_argument("--output", type=Path, required=True)
    calibration_validate.add_argument("--corpus", type=Path, required=True)
    calibration_validate.add_argument("--index", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "smoke":
        report = run_cpu_smoke(args.output, steps=args.steps)
    elif args.command == "validate":
        report = validate_cpu_smoke(args.report, exact_replay=args.exact_replay)
    elif args.command == "pilot":
        report = run_real_corpus_pilot(
            args.corpus,
            args.index,
            args.output,
            config=RealCorpusPilotConfig(steps=args.steps, eval_samples_per_split=args.eval_samples),
        )
    elif args.command == "validate-pilot":
        report = validate_real_corpus_pilot(
            args.report,
            corpus_root=args.corpus,
            index_root=args.index,
            exact_replay=args.exact_replay,
        )
    elif args.command in {"calibration-worker", "calibrate"}:
        config = CalibrationConfig(
            steps=args.steps,
            validation_batch_size=args.validation_batch_size,
            test_batch_size=args.test_batch_size,
            model=LocatorModelConfig(base_channels=args.base_channels),
        )
        if args.command == "calibration-worker":
            report = run_calibration_worker(
                args.corpus,
                args.index,
                args.output,
                config=config,
            )
        else:
            report = supervise_calibration(
                args.corpus,
                args.index,
                args.output,
                config=config,
                max_attempts=args.max_attempts,
                timeout_seconds=args.timeout_seconds,
            )
    else:
        report = validate_supervised_calibration(
            args.output,
            corpus_root=args.corpus,
            index_root=args.index,
        )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
