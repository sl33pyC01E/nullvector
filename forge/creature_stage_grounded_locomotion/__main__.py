from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contract import source_sha256
from .review import DEFAULT_OUTPUT, build_review, validate_review


def main() -> int:
    parser = argparse.ArgumentParser(description="NULLVECTOR grounded locomotion authority")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("source-info")
    build = sub.add_parser("build-review")
    build.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    build.add_argument("--record-visual-inspection", action="store_true")
    validate = sub.add_parser("validate-review")
    validate.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    validate.add_argument("--no-replay", action="store_true")
    args = parser.parse_args()
    if args.command == "source-info":
        result = {"passed": True, "source_sha256": source_sha256()}
    elif args.command == "build-review":
        result = build_review(args.output, visually_inspected=args.record_visual_inspection)
    else:
        result = validate_review(args.output, replay=not args.no_replay)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
