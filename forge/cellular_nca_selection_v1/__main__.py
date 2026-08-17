from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contract import DEFAULT_AUTHORITY, DEFAULT_OUTPUT
from .evaluation import evaluate, validate


def main() -> None:
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True)
    assess = sub.add_parser("evaluate"); assess.add_argument("--authority", type=Path, default=DEFAULT_AUTHORITY); assess.add_argument("--output", type=Path, default=DEFAULT_OUTPUT); assess.add_argument("--device", default="cuda")
    check = sub.add_parser("validate"); check.add_argument("output", type=Path, nargs="?", default=DEFAULT_OUTPUT)
    args = parser.parse_args(); result = evaluate(args.authority, args.output, device_name=args.device) if args.command == "evaluate" else validate(args.output)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
