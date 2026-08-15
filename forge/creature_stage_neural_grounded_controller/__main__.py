from __future__ import annotations

import argparse
import json
from pathlib import Path

from .training import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train neural muscle/contact controller over grounded physics")
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--updates", type=int, default=None); parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); report = train(args.output, updates=args.updates, device=args.device)
    print(json.dumps({"updates": report["updates"], "validation": report["validation"], "checkpoint": report["checkpoint"], "semantic_sha256": report["semantic_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
