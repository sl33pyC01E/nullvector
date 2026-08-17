from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluate import evaluate_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode and audit a trained topology-prior v2 checkpoint.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--samples", type=int, choices=(6,24), default=6)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--raw-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(evaluate_checkpoint(args.checkpoint, args.output, device_name=args.device, samples=args.samples, offset=args.offset, limit=args.limit, compile_maps=not args.raw_only), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
