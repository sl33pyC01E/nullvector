from __future__ import annotations

import argparse
import json

from .release import build, validate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "validate"))
    args = parser.parse_args()
    print(json.dumps(build() if args.command == "build" else validate(), indent=2))
