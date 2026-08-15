from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contract import DEFAULT_OUTPUT, DEFAULT_TEACHER, CellularMotionTransformerConfig, source_sha256
from .model import CellularMotionTransformer
from .training import prepare_production, run_cpu_smoke, train_segment, validate_cpu_smoke


def main() -> int:
    parser = argparse.ArgumentParser(description="NULLVECTOR cellular motion transformer")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("model-info")
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--output", type=Path, required=True)
    smoke.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    smoke.add_argument("--steps", type=int, default=4)
    validate = sub.add_parser("validate-smoke")
    validate.add_argument("--output", type=Path, required=True)
    prepare = sub.add_parser("prepare-production")
    prepare.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    prepare.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    segment = sub.add_parser("segment")
    segment.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    segment.add_argument("--end-step", type=int, required=True)
    args = parser.parse_args()
    if args.command == "model-info":
        model = CellularMotionTransformer()
        result = {
            "passed": True,
            "source_sha256": source_sha256(),
            "parameters": model.parameter_count,
            "config": CellularMotionTransformerConfig().to_dict(),
        }
    elif args.command == "smoke":
        result = run_cpu_smoke(args.output, teacher_root=args.teacher, steps=args.steps)
    elif args.command == "validate-smoke":
        result = validate_cpu_smoke(args.output)
    elif args.command == "prepare-production":
        result = prepare_production(args.output, teacher_root=args.teacher)
    else:
        result = train_segment(args.output, end_step=args.end_step)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
