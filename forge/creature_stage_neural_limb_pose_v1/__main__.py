from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contract import TrainingConfig
from .training import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train neural inverse-muscle limb pose driver")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=TrainingConfig().updates)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    report = train(args.output, updates=args.updates, device=args.device)
    print(json.dumps({"status": report["status"], "metrics": report["metrics"], "checkpoint": report["checkpoint"]}, indent=2))


if __name__ == "__main__":
    main()
