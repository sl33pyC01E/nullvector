from __future__ import annotations

import argparse
import json
from pathlib import Path

from .release import DEFAULT_OUTPUT, build, validate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("build", "validate")); parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT); args = parser.parse_args(argv)
    print(json.dumps(build(args.output) if args.command == "build" else validate(args.output), indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
