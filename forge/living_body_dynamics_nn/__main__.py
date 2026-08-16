from __future__ import annotations

import argparse
from pathlib import Path

from .contract import TrainingPlan
from .evaluation import evaluate
from .training import train_segment


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train-segment")
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--steps", type=int, default=250)
    train.add_argument("--batch-size", type=int, default=8)
    evaluation = subparsers.add_parser("evaluate")
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "train-segment":
        print(train_segment(args.output, TrainingPlan(segment_steps=args.steps, batch_size=args.batch_size)))
    else:
        print(evaluate(args.checkpoint, args.output))


if __name__ == "__main__":
    main()
