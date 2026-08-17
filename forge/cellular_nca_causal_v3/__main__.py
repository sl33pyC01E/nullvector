from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contract import DEFAULT_OUTPUT
from .evaluation import evaluate
from .training import run_supervisor, train_segment


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train"); train.add_argument("--output", type=Path, default=DEFAULT_OUTPUT); train.add_argument("--steps", type=int, default=256); train.add_argument("--segment-steps", type=int, default=64); train.add_argument("--batch-size", type=int, default=4)
    segment = sub.add_parser("segment"); segment.add_argument("--output", type=Path, required=True); segment.add_argument("--end-step", type=int, required=True)
    assess = sub.add_parser("evaluate"); assess.add_argument("--output", type=Path, default=DEFAULT_OUTPUT); assess.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.command == "train": result = run_supervisor(args.output, total_steps=args.steps, segment_steps=args.segment_steps, batch_size=args.batch_size)
    elif args.command == "segment": result = {"checkpoint": str(train_segment(args.output, args.end_step))}
    else: result = evaluate(args.output, device_name=args.device)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
