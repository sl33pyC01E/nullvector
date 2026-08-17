from __future__ import annotations

import argparse
import json

from .contract import TrainingPlan
from .training import evaluate, train


def main():
    parser = argparse.ArgumentParser(description="Train the recurrent world student")
    sub = parser.add_subparsers(dest="command", required=True)
    train_parser = sub.add_parser("train"); train_parser.add_argument("--updates", type=int, default=1500); train_parser.add_argument("--segment", type=int, default=250)
    sub.add_parser("evaluate")
    args = parser.parse_args()
    result = train(plan=TrainingPlan(total_updates=args.updates, segment_updates=args.segment)) if args.command == "train" else evaluate()
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
