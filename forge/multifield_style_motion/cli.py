from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .compiler import (
    compile_family_to_destination,
    compile_motion_style_bank,
    compiler_source_hash,
)
from .replay import replay_family_payload, replay_motion_style_bank


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m forge.multifield_style_motion",
        description="Deterministic motion-coherent categorical sprite presentation compiler.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    compile_parser = subparsers.add_parser("compile", help="Compile the full 520-clip bank")
    compile_parser.add_argument("asset_index", type=Path)
    compile_parser.add_argument("destination", type=Path)
    compile_parser.add_argument("--ffmpeg", type=Path, default=None)

    replay_parser = subparsers.add_parser("replay", help="Replay and verify every output byte")
    replay_parser.add_argument("manifest", type=Path)
    replay_parser.add_argument("--report", type=Path, default=None)

    family_parser = subparsers.add_parser("family-worker", help=argparse.SUPPRESS)
    family_parser.add_argument("asset_index", type=Path)
    family_parser.add_argument("destination", type=Path)
    family_parser.add_argument("family")

    family_replay_parser = subparsers.add_parser("family-replay-worker", help=argparse.SUPPRESS)
    family_replay_parser.add_argument("asset_index", type=Path)
    family_replay_parser.add_argument("output_root", type=Path)
    family_replay_parser.add_argument("family")

    subparsers.add_parser("source-hash", help="Print the bound compiler source SHA-256")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "compile":
        result = compile_motion_style_bank(
            arguments.asset_index,
            arguments.destination,
            ffmpeg_executable=arguments.ffmpeg,
        )
        print(json.dumps({
            "status": result["status"],
            "clip_count": result["clip_count"],
            "frame_count": result["frame_count"],
        }, sort_keys=True))
    elif arguments.command == "replay":
        result = replay_motion_style_bank(arguments.manifest, report_path=arguments.report)
        print(json.dumps(result, sort_keys=True))
    elif arguments.command == "family-worker":
        result = compile_family_to_destination(
            arguments.asset_index,
            arguments.destination,
            arguments.family,
        )
        print(json.dumps(result, sort_keys=True))
    elif arguments.command == "family-replay-worker":
        result = replay_family_payload(
            arguments.asset_index,
            arguments.output_root,
            arguments.family,
        )
        print(json.dumps(result, sort_keys=True))
    elif arguments.command == "source-hash":
        print(compiler_source_hash())
    else:  # pragma: no cover - argparse enforces the command set
        raise AssertionError(arguments.command)
    return 0
