from __future__ import annotations

import argparse
from pathlib import Path

from .cache import build_cache, validate_cache
from .contract import DEFAULT_CACHE, DEFAULT_OUTPUT
from .training import train


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-cache")
    build.add_argument("--output", type=Path, default=DEFAULT_CACHE)
    validate = commands.add_parser("validate-cache")
    validate.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    fit = commands.add_parser("train")
    fit.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    fit.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.command == "build-cache":
        print(build_cache(args.output))
    elif args.command == "validate-cache":
        print(validate_cache(args.cache))
    else:
        print(train(args.output, cache=args.cache))


if __name__ == "__main__":
    main()
