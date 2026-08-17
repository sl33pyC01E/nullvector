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
    args = parser.parse_args()
    print(json.dumps(evaluate_checkpoint(args.checkpoint, args.output, device_name=args.device), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
