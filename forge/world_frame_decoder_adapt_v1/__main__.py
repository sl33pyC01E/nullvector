from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contract import DEFAULT_OUTPUT, TrainingPlan
from .training import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Adapt the frozen world-frame VAE decoder to cellular scenes.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--updates", type=int, default=TrainingPlan().total_updates)
    parser.add_argument("--batch-size", type=int, default=TrainingPlan().batch_size)
    args = parser.parse_args()
    plan = TrainingPlan(total_updates=args.updates, batch_size=args.batch_size)
    print(json.dumps(train(args.output, plan=plan), indent=2))


if __name__ == "__main__":
    main()
