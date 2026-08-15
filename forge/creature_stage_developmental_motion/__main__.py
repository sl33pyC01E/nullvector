from __future__ import annotations

import argparse
import json
from pathlib import Path

from .compiler import build_candidate_corpus, validate_candidate_corpus
from .contract import DEFAULT_CORPUS, DEFAULT_OUTPUT, DEFAULT_PARENT, DEFAULT_PRIOR, DEFAULT_REVIEW, source_sha256
from .evaluation import evaluate_checkpoint, validate_evaluation
from .parent_prior import build_parent_prior, validate_parent_prior
from .smoke import run_parent_adapter_smoke, validate_parent_adapter_smoke
from .training import prepare_production, train_next_segment


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m forge.creature_stage_developmental_motion")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("source-info")
    build = subparsers.add_parser("build-corpus")
    build.add_argument("--output", type=Path, default=DEFAULT_CORPUS)
    build.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    validate = subparsers.add_parser("validate-corpus")
    validate.add_argument("--output", type=Path, default=DEFAULT_CORPUS)
    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    smoke.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    smoke.add_argument("--prior", type=Path, default=DEFAULT_PRIOR)
    smoke.add_argument("--output", type=Path, required=True)
    smoke.add_argument("--steps", type=int, default=12)
    verify = subparsers.add_parser("validate-smoke")
    verify.add_argument("--output", type=Path, required=True)
    prepare = subparsers.add_parser("prepare-production")
    prepare.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    prepare.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    prepare.add_argument("--prior", type=Path, default=DEFAULT_PRIOR)
    prepare.add_argument("--total-updates", type=int, default=2000)
    prepare.add_argument("--segment-updates", type=int, default=250)
    prepare.add_argument("--batch-size", type=int, default=10)
    train = subparsers.add_parser("train-next")
    train.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    evaluate.add_argument("--checkpoint", type=Path)
    evaluate.add_argument("--destination", type=Path)
    evaluate.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    validate_eval = subparsers.add_parser("validate-evaluation")
    validate_eval.add_argument("--output", type=Path, required=True)
    prior = subparsers.add_parser("build-parent-prior")
    prior.add_argument("--output", type=Path, default=DEFAULT_PRIOR)
    prior.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    prior.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    prior.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    validate_prior = subparsers.add_parser("validate-parent-prior")
    validate_prior.add_argument("--output", type=Path, default=DEFAULT_PRIOR)
    args = parser.parse_args()
    if args.command == "source-info":
        result = {"passed": True, "source_sha256": source_sha256()}
    elif args.command == "build-corpus":
        result = build_candidate_corpus(args.output, review=args.review)
    elif args.command == "validate-corpus":
        result = validate_candidate_corpus(args.output, replay=True)
    elif args.command == "smoke":
        result = run_parent_adapter_smoke(args.output, corpus=args.corpus, parent=args.parent, prior=args.prior, steps=args.steps)
    elif args.command == "validate-smoke":
        result = validate_parent_adapter_smoke(args.output, replay=True)
    elif args.command == "prepare-production":
        result = prepare_production(
            args.output, corpus=args.corpus, prior=args.prior, total_updates=args.total_updates,
            segment_updates=args.segment_updates, batch_size=args.batch_size,
        )
    elif args.command == "train-next":
        result = train_next_segment(args.output)
    elif args.command == "evaluate":
        result = evaluate_checkpoint(
            args.output, checkpoint=args.checkpoint,
            destination=args.destination, device=args.device,
        )
    elif args.command == "validate-evaluation":
        result = validate_evaluation(args.output)
    elif args.command == "build-parent-prior":
        result = build_parent_prior(args.output, corpus=args.corpus, parent=args.parent, device=args.device)
    else:
        result = validate_parent_prior(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
