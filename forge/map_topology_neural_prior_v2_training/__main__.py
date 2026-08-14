from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contract import FROZEN_LATENT_CORPUS_RELATIVE, PriorV2CalibrationConfig
from .training import run_segment, validate_segment


def main() -> int:
    parser = argparse.ArgumentParser(description="Segmented multiscale topology-prior v2 calibration")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("segment"); train.add_argument("--output", type=Path, required=True); train.add_argument("--resume", type=Path); train.add_argument("--corpus", type=Path, default=Path("outputs/map_decorator_corpus_v1")); train.add_argument("--latent-corpus", type=Path, default=Path(FROZEN_LATENT_CORPUS_RELATIVE)); train.add_argument("--device", choices=("cpu", "cuda"), default="cuda"); train.add_argument("--total-steps", type=int, default=24); train.add_argument("--steps-per-segment", type=int, default=4); train.add_argument("--width", type=int, default=48); train.add_argument("--validation-samples", type=int, choices=(6,48), default=48); train.add_argument("--test-samples", type=int, choices=(6,24), default=24)
    validate = sub.add_parser("validate"); validate.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "validate": result = validate_segment(args.output)
    else:
        config = PriorV2CalibrationConfig(total_steps=args.total_steps, steps_per_segment=args.steps_per_segment, width=args.width, validation_samples=args.validation_samples, test_samples=args.test_samples)
        result = run_segment(args.corpus, args.latent_corpus, args.output, config=config, resume=args.resume, device_name=args.device)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
