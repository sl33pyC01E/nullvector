from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contract import DEFAULT_OUTPUT, source_sha256
from .evaluation import evaluate_checkpoint, validate_evaluation
from .training import prepare_production, train_next_segment


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m forge.creature_stage_developmental_actuator_v3")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("source-info")
    prepare = commands.add_parser("prepare-production"); prepare.add_argument("--output", type=Path, default=DEFAULT_OUTPUT); prepare.add_argument("--total-updates", type=int, default=400); prepare.add_argument("--segment-updates", type=int, default=50); prepare.add_argument("--batch-size", type=int, default=5)
    train = commands.add_parser("train-next"); train.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    evaluate = commands.add_parser("evaluate"); evaluate.add_argument("--output", type=Path, default=DEFAULT_OUTPUT); evaluate.add_argument("--checkpoint", type=Path); evaluate.add_argument("--destination", type=Path); evaluate.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    validate = commands.add_parser("validate-evaluation"); validate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "source-info": result = {"passed": True, "source_sha256": source_sha256()}
    elif args.command == "prepare-production": result = prepare_production(args.output, total_updates=args.total_updates, segment_updates=args.segment_updates, batch_size=args.batch_size)
    elif args.command == "train-next": result = train_next_segment(args.output)
    elif args.command == "evaluate": result = evaluate_checkpoint(args.output, checkpoint=args.checkpoint, destination=args.destination, device=args.device)
    else: result = validate_evaluation(args.output)
    print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
