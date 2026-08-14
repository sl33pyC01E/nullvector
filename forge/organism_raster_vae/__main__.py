from __future__ import annotations

import argparse
from pathlib import Path

from .smoke import run_smoke, validate_smoke


def main() -> int:
    parser = argparse.ArgumentParser(description="Continuous cellular-organism VAE rasterizer")
    sub = parser.add_subparsers(dest="command", required=True)
    smoke = sub.add_parser("smoke"); smoke.add_argument("--output", type=Path, required=True); smoke.add_argument("--device", choices=("cpu", "cuda"), default="cpu"); smoke.add_argument("--steps", type=int, default=32)
    validate = sub.add_parser("validate"); validate.add_argument("output", type=Path)
    args = parser.parse_args()
    result = validate_smoke(args.output) if args.command == "validate" else run_smoke(args.output, device_name=args.device, steps=args.steps)
    print(result)
    return 0


if __name__ == "__main__": raise SystemExit(main())
