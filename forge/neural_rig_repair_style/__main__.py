from __future__ import annotations

import argparse
from pathlib import Path

from .authority import DEFAULT_REPAIR_BANK
from .pilot import DEFAULT_OUTPUT, compile_pilot


def main() -> int:
    parser = argparse.ArgumentParser(description="Export styled repaired neural rig animation")
    subparsers = parser.add_subparsers(dest="command", required=True)
    pilot = subparsers.add_parser("pilot", help="compile the five-family visual pilot")
    pilot.add_argument("--bank", type=Path, default=DEFAULT_REPAIR_BANK)
    pilot.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = compile_pilot(args.output, bank_path=args.bank)
    print(
        "REPAIR_STYLE_PILOT_OK",
        manifest["counts"]["identity_count"],
        manifest["counts"]["clip_count"],
        manifest["counts"]["frame_count"],
        manifest["pilot_sha256"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
