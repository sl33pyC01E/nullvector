from __future__ import annotations

import argparse
from pathlib import Path

from .evaluation import evaluate
from .training import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Neural grounded cellular locomotion")
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train"); fit.add_argument("--output", type=Path, required=True); fit.add_argument("--updates", type=int, default=None); fit.add_argument("--batch-size", type=int, default=None); fit.add_argument("--device", default="cuda")
    check = commands.add_parser("evaluate"); check.add_argument("--checkpoint", type=Path, required=True); check.add_argument("--output", type=Path, required=True); check.add_argument("--device", default="cuda"); check.add_argument("--model-state", action="store_true"); check.add_argument("--visually-inspected", action="store_true")
    args = parser.parse_args()
    if args.command == "train": result = train(args.output, updates=args.updates, batch_size=args.batch_size, device=args.device)
    else: result = evaluate(args.checkpoint, args.output, device=args.device, ema=not args.model_state, visually_inspected=args.visually_inspected)
    print(result)


if __name__ == "__main__": main()
