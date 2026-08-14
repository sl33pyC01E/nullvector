from __future__ import annotations

import argparse
import json
from pathlib import Path

from .compiler import build_bank, replay_bank, validate_bank
from .contract import DEFAULT_OUTPUT, DEFAULT_SOURCE


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compile deterministic cellular ontogeny programs")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build"); build.add_argument("--source", type=Path, default=DEFAULT_SOURCE); build.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    for name in ("validate", "replay"):
        command = sub.add_parser(name); command.add_argument("manifest", type=Path)
    args = parser.parse_args(argv)
    report = build_bank(args.source, args.output) if args.command == "build" else (validate_bank(args.manifest) if args.command == "validate" else replay_bank(args.manifest))
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
