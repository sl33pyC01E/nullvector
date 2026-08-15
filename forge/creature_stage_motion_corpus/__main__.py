from __future__ import annotations

import argparse
import json

from .validation import assert_valid_motion_corpus


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a NULLVECTOR cellular motion teacher corpus"
    )
    parser.add_argument("corpus")
    args = parser.parse_args()
    print(json.dumps(assert_valid_motion_corpus(args.corpus), indent=2))


if __name__ == "__main__":
    main()
