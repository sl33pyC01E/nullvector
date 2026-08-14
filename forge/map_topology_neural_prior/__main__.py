from __future__ import annotations

import argparse
import json
from pathlib import Path

from .smoke import build_smoke, validate_smoke


def main() -> int:
    parser = argparse.ArgumentParser(description="Masked latent topology-prior foundation")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("smoke", help="Build an immutable CPU-only masked-prior smoke")
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--corpus", type=Path, default=Path("outputs/map_decorator_corpus_v1"))
    validate = subparsers.add_parser("validate", help="Replay an immutable masked-prior smoke")
    validate.add_argument("--output", type=Path, required=True)
    validate.add_argument("--corpus", type=Path, default=Path("outputs/map_decorator_corpus_v1"))
    args = parser.parse_args()
    result = build_smoke(args.output, corpus_root=args.corpus) if args.command == "smoke" else validate_smoke(args.output, corpus_root=args.corpus)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

