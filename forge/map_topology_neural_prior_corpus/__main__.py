from __future__ import annotations

import argparse
import json
from pathlib import Path

from .shard import build_shard, validate_shard
from .supervisor import build_corpus, validate_corpus


def main() -> int:
    parser = argparse.ArgumentParser(description="Frozen neural map latent-token corpus")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--corpus", type=Path, default=Path("outputs/map_decorator_corpus_v1"))
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--workers", type=int, default=2)
    build.add_argument("--timeout-seconds", type=int, default=180)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--corpus", type=Path, default=Path("outputs/map_decorator_corpus_v1"))
    validate.add_argument("--output", type=Path, required=True)
    worker = subparsers.add_parser("worker")
    worker.add_argument("--mode", choices=("build", "validate"), required=True)
    worker.add_argument("--corpus", type=Path, required=True)
    worker.add_argument("--destination", type=Path, required=True)
    worker.add_argument("--shard-id", required=True)
    args = parser.parse_args()
    if args.command == "build":
        result = build_corpus(args.corpus, args.output, workers=args.workers, timeout_seconds=args.timeout_seconds)
    elif args.command == "validate":
        result = validate_corpus(args.corpus, args.output)
    elif args.mode == "build":
        result = build_shard(args.corpus, args.destination, args.shard_id)
    else:
        result = validate_shard(args.corpus, args.destination, replay_source=True)
    if args.command == "worker":
        result = {"status": "passed", "mode": args.mode, "shard_id": args.shard_id, "manifest_sha256": result["manifest_sha256"]}
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

