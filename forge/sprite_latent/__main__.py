from __future__ import annotations

import argparse
import json
from pathlib import Path

from .smoke import run_cpu_smoke, validate_smoke_output


def main() -> int:
    parser = argparse.ArgumentParser(description="Build or validate the semantic sprite FSQ foundation smoke.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--output", type=Path, required=True)
    smoke.add_argument("--continuous-steps", type=int, default=12)
    smoke.add_argument("--quantized-steps", type=int, default=36)
    validate = subparsers.add_parser("validate")
    validate.add_argument("manifest", type=Path)
    args = parser.parse_args()
    if args.command == "smoke":
        result = run_cpu_smoke(
            args.output,
            continuous_steps=args.continuous_steps,
            quantized_steps=args.quantized_steps,
        )
    else:
        result = validate_smoke_output(args.manifest)
    print(
        json.dumps(
            {
                "status": result["status"],
                "scope": result["scope"],
                "manifest_sha256": result["manifest_sha256"],
                "aligned_tuple_accuracy": result["reconstruction"]["aligned_tuple_accuracy"],
                "unique_code_count": result["reconstruction"]["unique_code_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
