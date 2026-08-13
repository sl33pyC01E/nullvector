from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contract import DEFAULT_OUTPUT
from .pilot import compile_production_pilot, validate_production_pilot


def main() -> int:
    parser = argparse.ArgumentParser(description="Production neural latent sprite genetics")
    commands = parser.add_subparsers(dest="command", required=True)
    compile_parser = commands.add_parser("compile"); compile_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    validate_parser = commands.add_parser("validate"); validate_parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    result = compile_production_pilot(args.output) if args.command == "compile" else validate_production_pilot(args.manifest)
    print(json.dumps({"status": result["status"], "counts": result["counts"], "bank_sha256": result["bank_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__": raise SystemExit(main())
