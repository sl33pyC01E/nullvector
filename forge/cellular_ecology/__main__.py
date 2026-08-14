from __future__ import annotations

import argparse
import json
from pathlib import Path

from .compiler import build_bank, replay_bank, validate_bank
from .contract import DEFAULT_MAP_ROOT, DEFAULT_OUTPUT


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build deterministic organism ecology fields")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--maps", type=Path, default=DEFAULT_MAP_ROOT)
    build.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    check = sub.add_parser("validate")
    check.add_argument("manifest", type=Path)
    replay = sub.add_parser("replay")
    replay.add_argument("manifest", type=Path)
    args = parser.parse_args(argv)
    if args.command == "build":
        report = build_bank(args.maps, args.output)
    elif args.command == "validate":
        report = validate_bank(args.manifest)
    else:
        report = replay_bank(args.manifest)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
