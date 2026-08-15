from __future__ import annotations

import argparse
from pathlib import Path

from .training import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the runtime-honest cyclic grounded motion successor")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(train(args.output, updates=args.updates, batch_size=args.batch_size, device=args.device))


if __name__ == "__main__":
    main()
