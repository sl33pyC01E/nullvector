from __future__ import annotations

import argparse
import json

from .validation import assert_valid_motion_audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a NULLVECTOR motion audit")
    parser.add_argument("audit")
    args = parser.parse_args()
    print(json.dumps(assert_valid_motion_audit(args.audit), indent=2))


if __name__ == "__main__":
    main()
