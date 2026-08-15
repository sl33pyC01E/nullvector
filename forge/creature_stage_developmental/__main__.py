from __future__ import annotations

import argparse
import json
from pathlib import Path

from .review import publish_review, validate_review


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m forge.creature_stage_developmental")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-review")
    build.add_argument("--output", type=Path, required=True)
    validate = subparsers.add_parser("validate-review")
    validate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = publish_review(args.output) if args.command == "build-review" else validate_review(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
