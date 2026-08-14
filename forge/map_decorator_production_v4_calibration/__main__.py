from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..map_decorator_ml.contract import ModelConfig
from .contract import CalibrationConfig
from .runner import run_worker, supervise, validate_supervised


def main() -> int:
    parser = argparse.ArgumentParser(description="Bounded proposal-aware map decorator v4 CUDA calibration")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("worker", "calibrate"):
        command = subparsers.add_parser(name)
        command.add_argument("--corpus", type=Path, required=True)
        command.add_argument("--index", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--steps", type=int, default=100)
        command.add_argument("--validation-batch-size", type=int, default=4)
        command.add_argument("--test-batch-size", type=int, default=1)
        command.add_argument("--base-channels", type=int, default=48)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--corpus", type=Path, required=True)
    validate.add_argument("--index", type=Path, required=True)
    validate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "validate":
        result = validate_supervised(args.output, corpus_root=args.corpus, index_root=args.index)
    else:
        config = CalibrationConfig(
            steps=args.steps,
            validation_batch_size=args.validation_batch_size,
            test_batch_size=args.test_batch_size,
            core=ModelConfig(base_channels=args.base_channels, condition_channels=args.base_channels * 2),
        )
        if args.command == "worker":
            result = run_worker(args.corpus, args.index, args.output, config=config)
        else:
            result = supervise(args.corpus, args.index, args.output, config=config)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
