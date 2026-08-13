from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

from ..config import OUTPUT_DIR
from .benchmark import benchmark_checkpoint
from .calibration import calibrate_morphology_corpus
from .checkpoint import (
    CheckpointNotReady,
    CheckpointProvenanceError,
    load_multifield_checkpoint,
    snapshot_published_checkpoint,
)
from .pipeline import replay_generation_bank, write_generation_bank


def _checkpoint_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", type=Path, required=True, help="Immutable published checkpoint snapshot.")
    parser.add_argument("--corpus", type=Path, help="Optional corpus relocation; SHA-256 must match the checkpoint.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="Inference device (auto selects CUDA when available).")
    parser.add_argument(
        "--precision", choices=("auto", "fp32", "bf16", "fp16"), default="auto", help="Autocast precision; fp16 requires CUDA."
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict evaluation, generation, and exact replay for v2 sprites."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    calibration_parser = subparsers.add_parser(
        "calibrate",
        help="Prove hard validity gates on held-out authoritative corpus fields.",
    )
    calibration_parser.add_argument("--corpus", type=Path, required=True)
    calibration_parser.add_argument("--validation-fraction", type=float, default=0.08)
    calibration_parser.add_argument(
        "--split-seed", type=lambda value: int(value, 0), default=0x5A17
    )
    calibration_parser.add_argument(
        "--guide-policy", choices=("scaffold_only", "full_debug"), default="scaffold_only"
    )
    calibration_parser.add_argument("--guide-thicken-radius", type=int, default=1)
    calibration_parser.add_argument("--guide-channel-dropout", type=float, default=0.08)
    calibration_parser.add_argument("--guide-jitter-pixels", type=int, default=1)
    calibration_parser.add_argument("--output", type=Path)

    snapshot_parser = subparsers.add_parser(
        "snapshot",
        help="Atomically copy one stable published checkpoint away from a live run.",
    )
    snapshot_parser.add_argument("source", type=Path, help="Atomically published latest.pt or best.pt.")
    snapshot_parser.add_argument("destination", type=Path, help="New immutable snapshot path; never overwritten.")

    status_parser = subparsers.add_parser(
        "status", help="Verify that a complete checkpoint is safely published."
    )
    _checkpoint_arguments(status_parser)

    sample_parser = subparsers.add_parser(
        "sample", help="Generate immutable raw and bounded compiled sprite banks."
    )
    _checkpoint_arguments(sample_parser)
    sample_parser.add_argument(
        "--grid", choices=("fixed", "stratified", "exhaustive"), default="stratified", help="Recorded bank, 40-cell coverage grid, or 160-cell subtype/role cross."
    )
    sample_parser.add_argument("--samples-per-condition", type=int, default=1, help="Independent seeded variations per condition cell.")
    sample_parser.add_argument("--base-seed", type=lambda value: int(value, 0), help="Nonnegative 63-bit seed override; decimal or 0x-prefixed.")
    sample_parser.add_argument("--limit", type=int, help="Prefix limit for smoke evaluation only.")
    sample_parser.add_argument("--batch-size", type=int, default=8, help="Recorded reverse-diffusion batch size.")
    sample_parser.add_argument("--temperature", type=float, default=0.9, help="Positive categorical sampling temperature.")
    sample_parser.add_argument("--postprocess-max-delta", type=float, default=0.03, help="Maximum removable pixel fraction (0 through 0.10).")
    sample_parser.add_argument("--output-dir", type=Path, help="New empty immutable bank directory.")

    replay_parser = subparsers.add_parser(
        "replay", help="Regenerate a bank and require exact raw/compiled equality."
    )
    replay_parser.add_argument("manifest", type=Path, help="generation_manifest.json to reproduce exactly.")
    replay_parser.add_argument("--checkpoint", type=Path)
    replay_parser.add_argument("--corpus", type=Path)
    replay_parser.add_argument("--device", choices=("cpu", "cuda"))
    replay_parser.add_argument(
        "--precision", choices=("fp32", "bf16", "fp16")
    )
    replay_parser.add_argument("--report", type=Path, help="Optional atomic JSON replay report output.")

    benchmark_parser = subparsers.add_parser(
        "benchmark", help="Benchmark full-mask accuracy and raw reverse diffusion."
    )
    _checkpoint_arguments(benchmark_parser)
    benchmark_parser.add_argument(
        "--grid", choices=("fixed", "stratified", "exhaustive"), default="stratified", help="Condition grid for raw generation benchmarking."
    )
    benchmark_parser.add_argument("--samples-per-condition", type=int, default=2, help="Variations per condition for within-condition diversity.")
    benchmark_parser.add_argument("--generation-limit", type=int, help="Prefix limit for smoke benchmarks.")
    benchmark_parser.add_argument("--generation-batch-size", type=int, default=8, help="Reverse-diffusion batch size.")
    benchmark_parser.add_argument("--full-mask-examples", type=int, default=256, help="Stratified validation examples for full-mask metrics.")
    benchmark_parser.add_argument("--full-mask-batch-size", type=int, default=32, help="Full-mask forward batch size.")
    benchmark_parser.add_argument("--temperature", type=float, default=0.9, help="Positive categorical sampling temperature.")
    benchmark_parser.add_argument("--base-seed", type=lambda value: int(value, 0), help="Nonnegative 63-bit generation seed override; decimal or 0x-prefixed.")
    benchmark_parser.add_argument("--output", type=Path, help="Optional atomic benchmark JSON path.")
    return parser.parse_args(argv)


def _load(args: argparse.Namespace):
    return load_multifield_checkpoint(
        args.checkpoint,
        corpus_path=args.corpus,
        device=args.device,
        precision=args.precision,
    )


def _print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "calibrate":
            from ..multifield_data import GuidePolicy

            report = calibrate_morphology_corpus(
                args.corpus,
                validation_fraction=args.validation_fraction,
                split_seed=args.split_seed,
                guide_policy=GuidePolicy(
                    name=args.guide_policy,
                    thicken_radius=args.guide_thicken_radius,
                    training_channel_dropout=args.guide_channel_dropout,
                    training_jitter_pixels=args.guide_jitter_pixels,
                ),
                output_path=args.output,
            )
            _print(report)
            return 0 if report["hard_valid_rate"] == 1.0 else 6
        if args.command == "snapshot":
            _print(snapshot_published_checkpoint(args.source, args.destination))
            return 0
        if args.command == "replay":
            report = replay_generation_bank(
                args.manifest,
                checkpoint_path=args.checkpoint,
                corpus_path=args.corpus,
                device=args.device,
                precision=args.precision,
                report_path=args.report,
            )
            _print(report)
            return 0 if report["status"] == "exact" else 5

        bundle = _load(args)
        if args.command == "status":
            _print({"status": "ready", "provenance": bundle.provenance()})
            return 0
        if args.command == "sample":
            destination = args.output_dir
            if destination is None:
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                destination = OUTPUT_DIR / "multifield_generation" / stamp
            manifest = write_generation_bank(
                bundle,
                destination,
                mode=args.grid,
                samples_per_condition=args.samples_per_condition,
                base_seed=args.base_seed,
                limit=args.limit,
                batch_size=args.batch_size,
                temperature=args.temperature,
                max_postprocess_delta=args.postprocess_max_delta,
            )
            _print(
                {
                    "status": "ready",
                    "manifest": str((Path(destination).resolve() / "generation_manifest.json")),
                    "samples": manifest["grid"]["samples"],
                    "acceptance": manifest["validation"]["acceptance"]["overall"],
                    "contact_sheets": manifest["contact_sheets"],
                }
            )
            return 0
        if args.command == "benchmark":
            report = benchmark_checkpoint(
                bundle,
                grid_mode=args.grid,
                samples_per_condition=args.samples_per_condition,
                generation_limit=args.generation_limit,
                generation_batch_size=args.generation_batch_size,
                full_mask_examples=args.full_mask_examples,
                full_mask_batch_size=args.full_mask_batch_size,
                temperature=args.temperature,
                base_seed=args.base_seed,
                output_path=args.output,
            )
            _print(report)
            return 0
        raise AssertionError(f"Unhandled command {args.command!r}")
    except CheckpointNotReady as error:
        _print(error.report())
        return 3
    except CheckpointProvenanceError as error:
        _print({"status": "checkpoint_rejected", "reason": str(error)})
        return 4


if __name__ == "__main__":
    sys.exit(main())
