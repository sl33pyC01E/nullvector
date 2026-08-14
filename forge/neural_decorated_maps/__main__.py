from __future__ import annotations

import argparse
import json
from pathlib import Path

from .compiler import build_bank, validate_bank


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile accepted neural map decorations into native atlases")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("build", "validate"):
        command = subparsers.add_parser(name)
        command.add_argument("--selection-audit", type=Path, required=True)
        command.add_argument("--calibration", type=Path, required=True)
        command.add_argument("--corpus", type=Path, required=True)
        command.add_argument("--index", type=Path, required=True)
        command.add_argument("--maps", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        if name == "build": command.add_argument("--visually-inspected", action="store_true")
    args = parser.parse_args()
    kwargs = {
        "selection_audit": args.selection_audit,
        "calibration_root": args.calibration,
        "corpus_root": args.corpus,
        "index_root": args.index,
        "map_root": args.maps,
    }
    if args.command == "build":
        result = build_bank(output=args.output, visually_inspected=args.visually_inspected, **kwargs)
    else:
        result = validate_bank(args.output, **kwargs)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
