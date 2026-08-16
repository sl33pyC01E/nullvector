from __future__ import annotations

import argparse
import json
from pathlib import Path

from .corpus import build_encoded_corpus, validate_encoded_corpus


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--trajectories", type=Path, nargs="+", required=True)
    build.add_argument("--vae", type=Path, required=True)
    build.add_argument("--device", default="cuda")
    validate = sub.add_parser("validate")
    validate.add_argument("corpus", type=Path)
    args = parser.parse_args()
    report = build_encoded_corpus(args.output, args.trajectories, args.vae, device=args.device) if args.command == "build" else validate_encoded_corpus(args.corpus)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
