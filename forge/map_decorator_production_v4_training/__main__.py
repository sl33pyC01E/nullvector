from __future__ import annotations

import argparse
import json
from pathlib import Path

from .calibration import run_calibration, validate_calibration
from .contract import ResidualCalibrationConfig
from .smoke import run_smoke, validate_smoke


def main() -> int:
    parser = argparse.ArgumentParser(description="Map decorator v4 proposal-residual training foundation")
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke = subparsers.add_parser("smoke", help="Build the immutable two-step CPU resume proof")
    smoke.add_argument("--corpus", type=Path, required=True)
    smoke.add_argument("--index", type=Path, required=True)
    smoke.add_argument("--output", type=Path, required=True)
    validate = subparsers.add_parser("validate", help="Replay and validate an immutable smoke proof")
    validate.add_argument("--corpus", type=Path, required=True)
    validate.add_argument("--index", type=Path, required=True)
    validate.add_argument("--output", type=Path, required=True)
    calibrate = subparsers.add_parser("calibrate", help="Run a bounded CUDA BF16 residual calibration")
    calibrate.add_argument("--corpus", type=Path, required=True)
    calibrate.add_argument("--index", type=Path, required=True)
    calibrate.add_argument("--output", type=Path, required=True)
    calibrate.add_argument("--steps", type=int, default=64)
    validate_calibration_parser = subparsers.add_parser(
        "validate-calibration", help="Replay a CUDA calibration and validate all metrics"
    )
    validate_calibration_parser.add_argument("--corpus", type=Path, required=True)
    validate_calibration_parser.add_argument("--index", type=Path, required=True)
    validate_calibration_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "smoke":
        result = run_smoke(args.corpus, args.index, args.output)
    elif args.command == "validate":
        result = validate_smoke(args.output, corpus_root=args.corpus, index_root=args.index)
    elif args.command == "calibrate":
        result = run_calibration(
            args.corpus,
            args.index,
            args.output,
            config=ResidualCalibrationConfig(steps=args.steps),
        )
    else:
        result = validate_calibration(
            args.output,
            corpus_root=args.corpus,
            index_root=args.index,
            replay_metrics=True,
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
