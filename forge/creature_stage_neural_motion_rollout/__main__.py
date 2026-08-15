from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contract import DEFAULT_OUTPUT, DEFAULT_PARENT, source_sha256
from .evaluation import evaluate_checkpoint, validate_evaluation
from .training import prepare_production, run_cpu_smoke, train_segment, validate_cpu_smoke


def main() -> int:
    parser = argparse.ArgumentParser(description="NULLVECTOR prediction-fed cellular motion rollout trainer")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("source-info")
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--output", type=Path, required=True)
    smoke.add_argument("--steps", type=int, default=8)
    validate = sub.add_parser("validate-smoke")
    validate.add_argument("--output", type=Path, required=True)
    prepare = sub.add_parser("prepare-production")
    prepare.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    prepare.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    segment = sub.add_parser("segment")
    segment.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    segment.add_argument("--end-update", type=int, required=True)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--split", choices=("validation", "test"), default="validation")
    evaluate.add_argument("--motions", type=int, nargs="+", default=list(range(13)))
    evaluate.add_argument("--frames", type=int, default=72)
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--release-sealed-test", action="store_true")
    verify = sub.add_parser("validate-evaluation")
    verify.add_argument("--report", type=Path, required=True)
    verify.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    if args.command == "source-info":
        result = {"passed": True, "source_sha256": source_sha256()}
    elif args.command == "smoke":
        result = run_cpu_smoke(args.output, steps=args.steps)
    elif args.command == "validate-smoke":
        result = validate_cpu_smoke(args.output, replay=True)
    elif args.command == "prepare-production":
        result = prepare_production(args.output, parent_checkpoint=args.parent)
    elif args.command == "segment":
        result = train_segment(args.output, end_update=args.end_update)
    elif args.command == "evaluate":
        result = evaluate_checkpoint(
            args.checkpoint, args.output, split=args.split, motion_ids=args.motions,
            rollout_frames=args.frames, device=args.device,
            allow_sealed_test=args.release_sealed_test,
        )
    else:
        result = validate_evaluation(args.report, replay=args.replay)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
