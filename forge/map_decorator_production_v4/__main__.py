from __future__ import annotations

import argparse
import json
from pathlib import Path

from .smoke import build_smoke, validate_smoke
from .audit import audit_chunk, build_full_audit, validate_full_audit


def main() -> int:
    parser = argparse.ArgumentParser(description="Map decorator v4 public-entropy proposal substrate")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("smoke")
    build.add_argument("--corpus", type=Path, required=True)
    build.add_argument("--index", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--visually-inspected", action="store_true")
    validate = sub.add_parser("validate")
    validate.add_argument("--corpus", type=Path, required=True)
    validate.add_argument("--index", type=Path, required=True)
    validate.add_argument("--output", type=Path, required=True)
    worker = sub.add_parser("audit-worker")
    worker.add_argument("--corpus", type=Path, required=True)
    worker.add_argument("--index", type=Path, required=True)
    worker.add_argument("--output", type=Path, required=True)
    worker.add_argument("--chunk-index", type=int, required=True)
    audit = sub.add_parser("audit-corpus")
    audit.add_argument("--corpus", type=Path, required=True)
    audit.add_argument("--index", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    audit_validate = sub.add_parser("validate-audit")
    audit_validate.add_argument("--corpus", type=Path, required=True)
    audit_validate.add_argument("--index", type=Path, required=True)
    audit_validate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "smoke":
        result = build_smoke(
            args.corpus,
            args.index,
            args.output,
            visually_inspected=args.visually_inspected,
        )
    elif args.command == "validate":
        result = validate_smoke(args.output, corpus_root=args.corpus, index_root=args.index)
    elif args.command == "audit-worker":
        result = audit_chunk(
            args.corpus,
            args.index,
            args.output,
            chunk_index=args.chunk_index,
        )
    elif args.command == "audit-corpus":
        result = build_full_audit(args.corpus, args.index, args.output)
    else:
        result = validate_full_audit(args.output, corpus_root=args.corpus, index_root=args.index)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
