from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contract import DEFAULT_AUTHORITY
from .evaluation import audit, validate


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the causal NCA on native living bodies.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("destination", type=Path)
    audit_parser.add_argument("--authority", type=Path, default=DEFAULT_AUTHORITY)
    audit_parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("path", type=Path)
    args = parser.parse_args()
    result = (
        audit(args.destination, authority=args.authority, device=args.device)
        if args.command == "audit"
        else validate(args.path)
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
