from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .compiler import compile_generation_bank, compiler_source_hash
from .io import write_json_new
from .procedural import (
    compile_procedural_reference_bank,
    replay_procedural_reference_bank,
)
from .replay import replay_style_bank


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m forge.multifield_style",
        description="Deterministic CPU-only presentation compiler for categorical sprites.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    compile_parser = subparsers.add_parser(
        "compile", help="Compile an immutable neural generation bank."
    )
    compile_parser.add_argument("generation_manifest", type=Path)
    compile_parser.add_argument("destination", type=Path)
    compile_parser.add_argument("--map-art-root", type=Path)

    replay_parser = subparsers.add_parser(
        "replay", help="Strictly replay a derived neural style bank."
    )
    replay_parser.add_argument("style_manifest", type=Path)
    replay_parser.add_argument("--map-art-root", type=Path)
    replay_parser.add_argument("--report", type=Path)

    reference_parser = subparsers.add_parser(
        "procedural-reference",
        help="Compile a clearly labeled non-neural five-family reference bank.",
    )
    reference_parser.add_argument("procedural_manifest", type=Path)
    reference_parser.add_argument("destination", type=Path)
    reference_parser.add_argument("--map-art-root", type=Path)

    reference_replay_parser = subparsers.add_parser(
        "replay-procedural-reference",
        help="Strictly replay a five-family procedural reference bank.",
    )
    reference_replay_parser.add_argument("procedural_reference_manifest", type=Path)
    reference_replay_parser.add_argument("--map-art-root", type=Path)
    reference_replay_parser.add_argument("--report", type=Path)

    subparsers.add_parser("source-hash", help="Print the current compiler source hash.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "compile":
        result = compile_generation_bank(
            arguments.generation_manifest,
            arguments.destination,
            map_art_root=arguments.map_art_root,
        )
        print(
            json.dumps(
                {
                    "manifest": str(Path(arguments.destination).resolve() / "style_manifest.json"),
                    "samples": result["sample_count"],
                    "compiler_source_sha256": result["compiler"]["source_sha256"],
                },
                sort_keys=True,
            )
        )
        return 0
    if arguments.command == "replay":
        result = replay_style_bank(
            arguments.style_manifest,
            map_art_root=arguments.map_art_root,
        )
        if arguments.report is not None:
            write_json_new(arguments.report, result)
        print(json.dumps(result, sort_keys=True))
        return 0 if result["passed"] else 1
    if arguments.command == "procedural-reference":
        result = compile_procedural_reference_bank(
            arguments.procedural_manifest,
            arguments.destination,
            map_art_root=arguments.map_art_root,
        )
        print(
            json.dumps(
                {
                    "manifest": str(
                        Path(arguments.destination).resolve()
                        / "procedural_reference_manifest.json"
                    ),
                    "samples": result["sample_count"],
                    "neural_output": result["neural_output"],
                    "compiler_source_sha256": result["compiler"]["source_sha256"],
                },
                sort_keys=True,
            )
        )
        return 0
    if arguments.command == "replay-procedural-reference":
        result = replay_procedural_reference_bank(
            arguments.procedural_reference_manifest,
            map_art_root=arguments.map_art_root,
        )
        if arguments.report is not None:
            write_json_new(arguments.report, result)
        print(json.dumps(result, sort_keys=True))
        return 0 if result["passed"] else 1
    if arguments.command == "source-hash":
        print(compiler_source_hash())
        return 0
    raise RuntimeError(f"Unhandled command: {arguments.command}")
