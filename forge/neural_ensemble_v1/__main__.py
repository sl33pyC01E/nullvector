from __future__ import annotations

import argparse
import json

from .release import build, validate


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or validate the factorized neural teacher ensemble")
    sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--no-loader-probes", action="store_true")
    sub.add_parser("validate")
    args = parser.parse_args()
    result = build(probe_loaders=not args.no_loader_probes) if args.command == "build" else validate()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
