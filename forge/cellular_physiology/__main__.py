from __future__ import annotations

import argparse
import json
from pathlib import Path

from .compiler import build_bank, replay_bank, validate_bank
from .contract import DEFAULT_OUTPUT, DEFAULT_SOURCE


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compile connected pixel-cell organ systems")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--organisms", type=Path, default=DEFAULT_SOURCE)
    build.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    for name in ("validate", "replay"):
        command = subparsers.add_parser(name)
        command.add_argument("manifest", type=Path)
    args = parser.parse_args(argv)
    if args.command == "build":
        report = build_bank(args.organisms, args.output)
    elif args.command == "validate":
        report = validate_bank(args.manifest)
    else:
        report = replay_bank(args.manifest)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
