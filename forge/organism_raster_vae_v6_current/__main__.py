from __future__ import annotations

import argparse
from pathlib import Path

from .contract import DEFAULT_OUTPUT, TrainingPlan
from .training import train_segment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("train-segment", nargs="?")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    print(train_segment(args.output, TrainingPlan(segment_steps=args.steps, batch_size=args.batch_size)))


if __name__ == "__main__":
    main()
