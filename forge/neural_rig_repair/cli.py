from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .compiler import compile_repair_bank, prepare_repair_plans
from .constants import (
    DEFAULT_GENERATION_MANIFEST,
    DEFAULT_OUTPUT,
    DEFAULT_STYLE_MANIFEST,
    STRESS_MAX_ATTEMPTS,
    STRESS_TIMEOUT_SECONDS,
    STRESS_WORKERS,
)
from .replay import replay_repair_bank


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m forge.neural_rig_repair",
        description="Compile and exactly replay CPU-only neural rig repair-v2 plans.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "compile"):
        command = subparsers.add_parser(name)
        command.add_argument("--generation", type=Path, default=DEFAULT_GENERATION_MANIFEST)
        command.add_argument("--style", type=Path, default=DEFAULT_STYLE_MANIFEST)
        command.add_argument("--destination", type=Path, default=DEFAULT_OUTPUT)
        if name == "compile":
            command.add_argument("--workers", type=int, default=STRESS_WORKERS)
            command.add_argument(
                "--timeout-seconds", type=int, default=STRESS_TIMEOUT_SECONDS
            )
            command.add_argument(
                "--max-attempts", type=int, default=STRESS_MAX_ATTEMPTS
            )
            command.add_argument(
                "--skip-exact-replay",
                action="store_true",
                help="Build the bank but defer the independent 75,520-frame replay.",
            )
    replay = subparsers.add_parser("replay")
    replay.add_argument("manifest", type=Path)
    replay.add_argument("--report", type=Path)
    replay.add_argument(
        "--metadata-only",
        action="store_true",
        help="Inspect signed stress artifacts without independently rerendering frames.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        context, samples = prepare_repair_plans(
            args.generation, args.style, args.destination
        )
        payload = {
            "status": "prepared",
            "sample_count": len(samples),
            "rest_audit_sha256": context["rest_audit"]["rest_audit_sha256"],
            "destination": str(Path(args.destination).resolve()),
        }
    elif args.command == "compile":
        bank = compile_repair_bank(
            args.generation,
            args.style,
            args.destination,
            workers=args.workers,
            timeout_seconds=args.timeout_seconds,
            max_attempts=args.max_attempts,
            exact_replay=not args.skip_exact_replay,
        )
        payload = {
            "status": bank["status"],
            "sample_count": bank["build_contract"]["sample_count"],
            "clip_count": bank["build_contract"]["clip_count"],
            "frame_count": bank["build_contract"]["frame_count"],
            "bank_sha256": bank["bank_sha256"],
        }
    else:
        report = replay_repair_bank(
            args.manifest,
            report_path=args.report,
            rerun_motion=not args.metadata_only,
        )
        payload = {
            "status": report["status"],
            "mode": report["mode"],
            "sample_count": report["counts"]["sample_count"],
            "clip_count": report["counts"]["clip_count"],
            "frame_count": report["counts"]["frame_count"],
            "replay_sha256": report["replay_sha256"],
        }
    print(json.dumps(payload, sort_keys=True))
    return 0
