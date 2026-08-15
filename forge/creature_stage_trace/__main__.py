from __future__ import annotations

import argparse
import json

from .validation import assert_valid_trace


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail-closed validation for NULLVECTOR causal game traces."
    )
    parser.add_argument("trace", help="Path to a creature-stage trace JSON file")
    args = parser.parse_args()
    print(json.dumps(assert_valid_trace(args.trace), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

