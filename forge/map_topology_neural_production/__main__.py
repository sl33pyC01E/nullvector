from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contract import TopologyCodecCalibrationConfig
from .training import run_calibration, validate_calibration


def main() -> int:
    parser = argparse.ArgumentParser(description="Neural map-topology codec production calibration")
    subparsers = parser.add_subparsers(dest="command", required=True)
    calibrate = subparsers.add_parser("calibrate", help="Run a bounded CUDA BF16 representation calibration")
    calibrate.add_argument("--corpus", type=Path, default=Path("outputs/map_decorator_corpus_v1"))
    calibrate.add_argument("--output", type=Path, required=True)
    calibrate.add_argument("--steps", type=int, default=100)
    calibrate.add_argument("--validation-samples", type=int, default=48, choices=(6, 48))
    calibrate.add_argument("--test-samples", type=int, default=24, choices=(6, 24))
    validate = subparsers.add_parser("validate", help="Validate and optionally replay a calibration")
    validate.add_argument("--corpus", type=Path, default=Path("outputs/map_decorator_corpus_v1"))
    validate.add_argument("--output", type=Path, required=True)
    validate.add_argument("--no-metric-replay", action="store_true")
    args = parser.parse_args()
    if args.command == "calibrate":
        result = run_calibration(
            args.corpus,
            args.output,
            config=TopologyCodecCalibrationConfig(
                steps=args.steps,
                validation_samples=args.validation_samples,
                test_samples=args.test_samples,
            ),
        )
    else:
        result = validate_calibration(
            args.output,
            corpus_root=args.corpus,
            replay_metrics=not args.no_metric_replay,
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
