from __future__ import annotations

import argparse
from pathlib import Path

from .contract import DEFAULT_RELEASE
from .release import build_release, validate_release


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "validate"))
    parser.add_argument("--output", type=Path, default=DEFAULT_RELEASE)
    args = parser.parse_args()
    print(build_release(args.output) if args.command == "build" else validate_release(args.output))


if __name__ == "__main__":
    main()
