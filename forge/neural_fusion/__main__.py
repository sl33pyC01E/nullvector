from __future__ import annotations

import argparse
from pathlib import Path

from .pilot import DEFAULT_OUTPUT, compile_pilot


def main() -> int:
    parser = argparse.ArgumentParser(description="Neural sprite fusion and mutation forge")
    subparsers = parser.add_subparsers(dest="command", required=True)
    pilot = subparsers.add_parser("pilot")
    pilot.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = compile_pilot(args.output)
    print(
        "NEURAL_FUSION_PILOT_OK",
        manifest["counts"]["specimen_count"],
        manifest["counts"]["clip_count"],
        manifest["counts"]["frame_count"],
        manifest["bank_sha256"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
