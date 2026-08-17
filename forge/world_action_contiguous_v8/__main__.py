from __future__ import annotations

import argparse
import json

from .contract import DEFAULT_OUTPUT
from .corpus import build, validate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "validate"))
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build(args.output) if args.command == "build" else validate(args.output), indent=2))
