from __future__ import annotations

import argparse
import json
from pathlib import Path

from .build import build_world_bank, validate_world_bank
from .contract import DEFAULT_OUTPUT


def main() -> None:
    parser = argparse.ArgumentParser(description="Compose neural topology and accepted neural decoration.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    build.add_argument("--visually-inspected", action="store_true")
    validate = subparsers.add_parser("validate")
    validate.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_world_bank(args.output, visually_inspected=args.visually_inspected) if args.command == "build" else validate_world_bank(args.output)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
