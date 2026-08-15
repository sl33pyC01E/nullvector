from __future__ import annotations

import argparse
from pathlib import Path

from .training import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Continue the sealed cyclic grounded checkpoint")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=600)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(train(args.output, updates=args.updates, device=args.device))


if __name__ == "__main__":
    main()
