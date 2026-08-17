from __future__ import annotations

import argparse
import json

from .showcase import build


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the promoted continuous cellular VAE showcase")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(json.dumps(build(device=args.device), indent=2))


if __name__ == "__main__":
    main()
