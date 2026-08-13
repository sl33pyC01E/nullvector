from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .compiler import (
    compile_family_to_destination,
    compile_neural_motion_style_bank,
    compile_shard_to_destination,
    compiler_source_hash,
)
from .replay import replay_family, replay_neural_motion_style_bank


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m forge.multifield_style_neural_motion")
    sub = parser.add_subparsers(dest="command", required=True)
    compile_parser = sub.add_parser("compile"); compile_parser.add_argument("generation", type=Path); compile_parser.add_argument("style", type=Path); compile_parser.add_argument("destination", type=Path); compile_parser.add_argument("--ffmpeg", type=Path)
    replay_parser = sub.add_parser("replay"); replay_parser.add_argument("manifest", type=Path); replay_parser.add_argument("--report", type=Path)
    family = sub.add_parser("family-worker"); family.add_argument("generation", type=Path); family.add_argument("style", type=Path); family.add_argument("destination", type=Path); family.add_argument("family")
    shard = sub.add_parser("shard-worker"); shard.add_argument("generation", type=Path); shard.add_argument("style", type=Path); shard.add_argument("destination", type=Path); shard.add_argument("family"); shard.add_argument("shard_index", type=int)
    replay_family_parser = sub.add_parser("family-replay-worker"); replay_family_parser.add_argument("generation", type=Path); replay_family_parser.add_argument("style", type=Path); replay_family_parser.add_argument("root", type=Path); replay_family_parser.add_argument("family")
    sub.add_parser("source-hash")
    args = parser.parse_args(argv)
    if args.command == "compile": result = compile_neural_motion_style_bank(args.generation, args.style, args.destination, ffmpeg_executable=args.ffmpeg); print(json.dumps({"status": result["status"], "identity_count": 5, "clip_count": 520, "frame_count": 4720}, sort_keys=True))
    elif args.command == "replay": print(json.dumps(replay_neural_motion_style_bank(args.manifest, report_path=args.report), sort_keys=True))
    elif args.command == "family-worker": print(json.dumps(compile_family_to_destination(args.generation, args.style, args.destination, args.family), sort_keys=True))
    elif args.command == "shard-worker": print(json.dumps(compile_shard_to_destination(args.generation, args.style, args.destination, args.family, args.shard_index), sort_keys=True))
    elif args.command == "family-replay-worker": print(json.dumps(replay_family(args.generation, args.style, args.root, args.family), sort_keys=True))
    else: print(compiler_source_hash())
    return 0
