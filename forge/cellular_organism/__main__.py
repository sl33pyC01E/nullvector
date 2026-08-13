from __future__ import annotations

import argparse
import json
from pathlib import Path

from .compiler import build_bank, replay_bank, validate_bank


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compile destructible pixel-cell organisms")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--generation", type=Path, required=True)
    build.add_argument("--style", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("manifest", type=Path)
    replay = sub.add_parser("replay")
    replay.add_argument("manifest", type=Path)
    args = parser.parse_args(argv)
    if args.command == "build":
        report = build_bank(args.generation, args.style, args.output)
    elif args.command == "validate":
        report = validate_bank(args.manifest)
    else:
        report = replay_bank(args.manifest)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
