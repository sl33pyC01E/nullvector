from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contract import DEFAULT_CORPUS
from .dataset import validate_corpus
from .model import NeuralCellMotionUNet
from .supervisor import build_corpus_resilient, validate_corpus_resilient
from .training import run_cpu_smoke, validate_cpu_smoke


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build and validate the learned cellular-motion corpus."); commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-corpus"); build.add_argument("--output", type=Path, default=DEFAULT_CORPUS); build.add_argument("--identities-per-family", type=int, default=None, help="Balanced prefix per family for smoke corpora; omit to compile all 45 authoritative identities."); build.add_argument("--workers", type=int, default=2); build.add_argument("--max-attempts", type=int, default=3); build.add_argument("--timeout-seconds", type=int, default=600); build.add_argument("--recover-from", action="append", type=Path, default=[])
    validate = commands.add_parser("validate-corpus"); validate.add_argument("--output", type=Path, default=DEFAULT_CORPUS); validate.add_argument("--replay", action="store_true"); validate.add_argument("--sample-id", action="append", default=[], help=argparse.SUPPRESS); validate.add_argument("--workers", type=int, default=2); validate.add_argument("--max-attempts", type=int, default=3); validate.add_argument("--timeout-seconds", type=int, default=600)
    smoke = commands.add_parser("smoke-train"); smoke.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS); smoke.add_argument("--output", type=Path, required=True); smoke.add_argument("--steps", type=int, default=3)
    validate_smoke = commands.add_parser("validate-smoke"); validate_smoke.add_argument("--output", type=Path, required=True)
    commands.add_parser("model-info"); args = parser.parse_args(argv)
    if args.command == "build-corpus": result = build_corpus_resilient(args.output, identities_per_family=args.identities_per_family, workers=args.workers, max_attempts=args.max_attempts, timeout_seconds=args.timeout_seconds, recover_from=args.recover_from)
    elif args.command == "validate-corpus": result = validate_corpus(args.output, replay=args.replay, record_ids=set(args.sample_id)) if args.sample_id else validate_corpus_resilient(args.output, replay=args.replay, workers=args.workers, max_attempts=args.max_attempts, timeout_seconds=args.timeout_seconds)
    elif args.command == "smoke-train": result = run_cpu_smoke(args.corpus, args.output, steps=args.steps)
    elif args.command == "validate-smoke": result = validate_cpu_smoke(args.output)
    else:
        model = NeuralCellMotionUNet(); result = {"parameters": model.parameter_count, "config": model.config.to_dict()}
    print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
