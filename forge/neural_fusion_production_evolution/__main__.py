from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contract import DEFAULT_FOUNDERS, DEFAULT_OUTPUT
from .evolution import compile_production_evolution, validate_production_evolution


def main() -> int:
    parser = argparse.ArgumentParser(description="Production neural-latent sprite evolution")
    commands = parser.add_subparsers(dest="command", required=True)
    compile_parser = commands.add_parser("compile")
    compile_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    compile_parser.add_argument("--founders", type=Path, default=DEFAULT_FOUNDERS)
    validate_parser = commands.add_parser("validate")
    validate_parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    result = (
        compile_production_evolution(args.output, founders_manifest=args.founders)
        if args.command == "compile"
        else validate_production_evolution(args.manifest)
    )
    print(json.dumps({"status": result["status"], "counts": result["counts"], "evolution_sha256": result["evolution_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
