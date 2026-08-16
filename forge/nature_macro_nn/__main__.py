from __future__ import annotations

import argparse
import json
from pathlib import Path

from .corpus import build_corpus, validate_corpus
from .training import train
from .contract import TrainingConfig


def main():
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-corpus"); build.add_argument("destination", type=Path); build.add_argument("--worlds", type=int, default=32); build.add_argument("--steps", type=int, default=48)
    validate = sub.add_parser("validate-corpus"); validate.add_argument("root", type=Path)
    fit = sub.add_parser("train"); fit.add_argument("corpus", type=Path); fit.add_argument("output", type=Path); fit.add_argument("--steps", type=int, default=2400); fit.add_argument("--batch-size", type=int, default=24); fit.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.command == "build-corpus": result = build_corpus(args.destination, worlds=args.worlds, steps=args.steps)
    elif args.command == "validate-corpus": result = validate_corpus(args.root)
    else: result = train(args.corpus, args.output, training=TrainingConfig(steps=args.steps, batch_size=args.batch_size), device=args.device)
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
