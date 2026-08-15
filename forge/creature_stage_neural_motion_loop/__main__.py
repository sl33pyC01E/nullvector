from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contract import source_sha256
from .smoke import run_cpu_smoke, validate_cpu_smoke


def main() -> int:
    parser = argparse.ArgumentParser(description="NULLVECTOR loop-aware cellular motion successor")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("source-info")
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--output", type=Path, required=True)
    smoke.add_argument("--steps", type=int, default=8)
    validate = sub.add_parser("validate-smoke")
    validate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "source-info":
        result = {"passed": True, "source_sha256": source_sha256()}
    elif args.command == "smoke":
        result = run_cpu_smoke(args.output, steps=args.steps)
    else:
        result = validate_cpu_smoke(args.output, replay=True)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
