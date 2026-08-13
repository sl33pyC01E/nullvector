from __future__ import annotations

import argparse
import json
from pathlib import Path

from .compiler import build_bank, replay_bank, validate_bank
from .contract import DEFAULT_OUTPUT, DEFAULT_SOURCE


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Breed structurally inherited pixel-cell organisms")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--parents", type=Path, default=DEFAULT_SOURCE)
    build.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    validate = commands.add_parser("validate")
    validate.add_argument("manifest", type=Path)
    replay = commands.add_parser("replay")
    replay.add_argument("manifest", type=Path)
    args = parser.parse_args(argv)
    if args.command == "build":
        result = build_bank(args.parents, args.output)
    elif args.command == "validate":
        result = validate_bank(args.manifest)
    else:
        result = replay_bank(args.manifest)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
