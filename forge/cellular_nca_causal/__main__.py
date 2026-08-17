from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contract import DEFAULT_OUTPUT, PARENT_OUTPUT
from .evaluation import evaluate, validate_output
from .training import prepare_training, run_supervisor, train_segment


def main() -> None:
    parser = argparse.ArgumentParser(description="Counterfactual organ-causality curriculum for the cellular NCA")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare"); prepare.add_argument("--output", type=Path, default=DEFAULT_OUTPUT); prepare.add_argument("--parent", type=Path, default=PARENT_OUTPUT); prepare.add_argument("--steps", type=int, default=512); prepare.add_argument("--segment-steps", type=int, default=128); prepare.add_argument("--batch-size", type=int, default=8); prepare.add_argument("--max-attempts", type=int, default=3)
    segment = sub.add_parser("segment"); segment.add_argument("--output", type=Path, required=True); segment.add_argument("--end-step", type=int, required=True)
    train = sub.add_parser("train"); train.add_argument("--output", type=Path, default=DEFAULT_OUTPUT); train.add_argument("--parent", type=Path, default=PARENT_OUTPUT); train.add_argument("--steps", type=int, default=512); train.add_argument("--segment-steps", type=int, default=128); train.add_argument("--batch-size", type=int, default=8); train.add_argument("--max-attempts", type=int, default=3)
    assess = sub.add_parser("evaluate"); assess.add_argument("--output", type=Path, default=DEFAULT_OUTPUT); assess.add_argument("--device", default="cpu")
    validate = sub.add_parser("validate"); validate.add_argument("--output", type=Path, default=DEFAULT_OUTPUT); validate.add_argument("--device", default="cpu"); validate.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare": result = prepare_training(args.output, parent_output=args.parent, total_steps=args.steps, segment_steps=args.segment_steps, batch_size=args.batch_size, max_attempts=args.max_attempts)
    elif args.command == "segment": result = train_segment(args.output, end_step=args.end_step)
    elif args.command == "train": result = run_supervisor(args.output, parent_output=args.parent, total_steps=args.steps, segment_steps=args.segment_steps, batch_size=args.batch_size, max_attempts=args.max_attempts)
    elif args.command == "evaluate": result = evaluate(args.output, device_name=args.device)
    else: result = validate_output(args.output, rerun_evaluation=not args.metadata_only, device_name=args.device)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__": main()
