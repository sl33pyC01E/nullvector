from __future__ import annotations

import argparse
from pathlib import Path

from .artifacts import finalize_output, validate_output
from .training import run_supervisor, train_segment


def main() -> int:
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train"); train.add_argument("--output", type=Path, required=True); train.add_argument("--steps", type=int, default=4096); train.add_argument("--segment-steps", type=int, default=512); train.add_argument("--batch-size", type=int, default=64); train.add_argument("--max-attempts", type=int, default=3)
    segment = sub.add_parser("segment"); segment.add_argument("--output", type=Path, required=True); segment.add_argument("--end-step", type=int, required=True)
    finalize = sub.add_parser("finalize"); finalize.add_argument("--output", type=Path, required=True)
    validate = sub.add_parser("validate"); validate.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "train": result = run_supervisor(args.output, total_steps=args.steps, segment_steps=args.segment_steps, batch_size=args.batch_size, max_attempts=args.max_attempts)
    elif args.command == "segment": result = train_segment(args.output, end_step=args.end_step)
    elif args.command == "finalize": result = finalize_output(args.output)
    else: result = validate_output(args.output)
    print(result); return 0


if __name__ == "__main__": raise SystemExit(main())
