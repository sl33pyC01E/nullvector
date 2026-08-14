from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audit import build_audit, validate_audit


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit protected rare proposals over a frozen v4 checkpoint")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("build", "validate"):
        command = subparsers.add_parser(name)
        command.add_argument("--calibration", type=Path, required=True)
        command.add_argument("--corpus", type=Path, required=True)
        command.add_argument("--index", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        if name == "build":
            command.add_argument("--visually-inspected", action="store_true")
    args = parser.parse_args()
    if args.command == "build":
        result = build_audit(
            args.calibration, args.corpus, args.index, args.output,
            visually_inspected=args.visually_inspected,
        )
    else:
        result = validate_audit(
            args.output,
            calibration_root=args.calibration,
            corpus_root=args.corpus,
            index_root=args.index,
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
