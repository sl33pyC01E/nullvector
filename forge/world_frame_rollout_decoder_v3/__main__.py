from __future__ import annotations

import argparse
from pathlib import Path

from .contract import DEFAULT_OUTPUT
from .training import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(train(args.output))


if __name__ == "__main__":
    main()
