from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contract import DEFAULT_OUTPUT, DEFAULT_TEACHER, CellularPhysiologyTransformerConfig, source_sha256
from .model import CellularPhysiologyTransformer
from .training import assert_training_window, prepare_production, run_cpu_smoke, validate_cpu_smoke
from .evaluation import evaluate_checkpoint, validate_evaluation


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
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--interventions", type=int, nargs="+", default=list(range(9)))
    evaluate.add_argument("--frames", type=int, default=180)
    evaluate.add_argument("--device", default="cpu")
    verify = commands.add_parser("validate-evaluation")
    verify.add_argument("--report", type=Path, required=True)
    verify.add_argument("--replay", action="store_true")
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
    elif args.command == "check-training-window":
        result = assert_training_window(args.output)
    elif args.command == "evaluate":
        result = evaluate_checkpoint(
            args.checkpoint, args.output, intervention_ids=args.interventions,
            rollout_frames=args.frames, device=args.device,
        )
    else:
        result = validate_evaluation(args.report, replay=args.replay)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
