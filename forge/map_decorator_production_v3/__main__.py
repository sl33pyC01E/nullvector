from __future__ import annotations

import argparse
import json
from pathlib import Path

from .smoke import run_cpu_smoke, validate_cpu_smoke


def main() -> int:
    parser = argparse.ArgumentParser(description="Map decorator v3 sparse-localization foundation")
    sub = parser.add_subparsers(dest="command", required=True)
    smoke = sub.add_parser("smoke"); smoke.add_argument("--output", type=Path, required=True); smoke.add_argument("--steps", type=int, default=4)
    validate = sub.add_parser("validate"); validate.add_argument("report", type=Path); validate.add_argument("--exact-replay", action="store_true")
    args = parser.parse_args()
    report = run_cpu_smoke(args.output, steps=args.steps) if args.command == "smoke" else validate_cpu_smoke(args.report, exact_replay=args.exact_replay)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
