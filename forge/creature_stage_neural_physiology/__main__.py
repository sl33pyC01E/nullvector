from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contract import DEFAULT_OUTPUT, DEFAULT_TEACHER, CellularPhysiologyTransformerConfig, source_sha256
from .model import CellularPhysiologyTransformer
from .training import assert_training_window, prepare_production, run_cpu_smoke, validate_cpu_smoke


def main() -> int:
    parser = argparse.ArgumentParser(description="NULLVECTOR native-cell physiology transformer")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("model-info")
    smoke = commands.add_parser("smoke")
    smoke.add_argument("--output", type=Path, required=True)
    smoke.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    smoke.add_argument("--steps", type=int, default=4)
    validate = commands.add_parser("validate-smoke")
    validate.add_argument("--output", type=Path, required=True)
    prepare = commands.add_parser("prepare-production")
    prepare.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    prepare.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    window = commands.add_parser("check-training-window")
    window.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.command == "model-info":
        model = CellularPhysiologyTransformer()
        result = {
            "passed": True, "source_sha256": source_sha256(),
            "parameters": model.parameter_count, "config": CellularPhysiologyTransformerConfig().to_dict(),
        }
    elif args.command == "smoke":
        result = run_cpu_smoke(args.output, teacher_root=args.teacher, steps=args.steps)
    elif args.command == "validate-smoke":
        result = validate_cpu_smoke(args.output)
    elif args.command == "prepare-production":
        result = prepare_production(args.output, teacher_root=args.teacher)
    else:
        result = assert_training_window(args.output)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
