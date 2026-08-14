from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contract import DEFAULT_OUTPUT, CellularNCAConfig
from .corpus import build_corpus, validate_corpus
from .evaluation import evaluate, validate_output
from .training import prepare_training, run_supervisor, train_segment


def main() -> None:
    parser = argparse.ArgumentParser(description="Neural cellular organism dynamics")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("corpus"); build.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    validate_c = sub.add_parser("validate-corpus"); validate_c.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    prepare = sub.add_parser("prepare"); prepare.add_argument("--output", type=Path, default=DEFAULT_OUTPUT); prepare.add_argument("--steps", type=int, default=2048); prepare.add_argument("--segment-steps", type=int, default=256); prepare.add_argument("--batch-size", type=int, default=12); prepare.add_argument("--width", type=int, default=256); prepare.add_argument("--depth", type=int, default=10)
    segment = sub.add_parser("segment"); segment.add_argument("--output", type=Path, required=True); segment.add_argument("--end-step", type=int, required=True)
    train = sub.add_parser("train"); train.add_argument("--output", type=Path, default=DEFAULT_OUTPUT); train.add_argument("--steps", type=int, default=2048); train.add_argument("--segment-steps", type=int, default=256); train.add_argument("--batch-size", type=int, default=12); train.add_argument("--width", type=int, default=256); train.add_argument("--depth", type=int, default=10); train.add_argument("--max-attempts", type=int, default=3)
    assess = sub.add_parser("evaluate"); assess.add_argument("--output", type=Path, default=DEFAULT_OUTPUT); assess.add_argument("--device", default="cuda")
    validate = sub.add_parser("validate"); validate.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.command == "corpus": result = build_corpus(args.output)
    elif args.command == "validate-corpus": result = validate_corpus(args.output)
    elif args.command == "prepare": result = prepare_training(args.output, total_steps=args.steps, segment_steps=args.segment_steps, batch_size=args.batch_size, config=CellularNCAConfig(width=args.width, depth=args.depth))
    elif args.command == "segment": result = train_segment(args.output, end_step=args.end_step)
    elif args.command == "train": result = run_supervisor(args.output, total_steps=args.steps, segment_steps=args.segment_steps, batch_size=args.batch_size, config=CellularNCAConfig(width=args.width, depth=args.depth), max_attempts=args.max_attempts)
    elif args.command == "evaluate": result = evaluate(args.output, device_name=args.device)
    else: result = validate_output(args.output)
    print(json.dumps(result if isinstance(result, dict) else {"result": str(result)}, indent=2, default=str))


if __name__ == "__main__": main()
