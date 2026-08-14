from __future__ import annotations

import argparse
import json
from pathlib import Path

from .smoke import run_cpu_smoke, validate_cpu_smoke
from .pilot import RealCorpusPilotConfig, run_real_corpus_pilot, validate_real_corpus_pilot


def main() -> int:
    parser = argparse.ArgumentParser(description="Map decorator v3 sparse-localization foundation")
    sub = parser.add_subparsers(dest="command", required=True)
    smoke = sub.add_parser("smoke"); smoke.add_argument("--output", type=Path, required=True); smoke.add_argument("--steps", type=int, default=4)
    validate = sub.add_parser("validate"); validate.add_argument("report", type=Path); validate.add_argument("--exact-replay", action="store_true")
    pilot = sub.add_parser("pilot")
    pilot.add_argument("--corpus", type=Path, required=True)
    pilot.add_argument("--index", type=Path, required=True)
    pilot.add_argument("--output", type=Path, required=True)
    pilot.add_argument("--steps", type=int, default=4)
    pilot.add_argument("--eval-samples", type=int, default=4)
    pilot_validate = sub.add_parser("validate-pilot")
    pilot_validate.add_argument("report", type=Path)
    pilot_validate.add_argument("--corpus", type=Path, required=True)
    pilot_validate.add_argument("--index", type=Path, required=True)
    pilot_validate.add_argument("--exact-replay", action="store_true")
    args = parser.parse_args()
    if args.command == "smoke":
        report = run_cpu_smoke(args.output, steps=args.steps)
    elif args.command == "validate":
        report = validate_cpu_smoke(args.report, exact_replay=args.exact_replay)
    elif args.command == "pilot":
        report = run_real_corpus_pilot(
            args.corpus,
            args.index,
            args.output,
            config=RealCorpusPilotConfig(steps=args.steps, eval_samples_per_split=args.eval_samples),
        )
    else:
        report = validate_real_corpus_pilot(
            args.report,
            corpus_root=args.corpus,
            index_root=args.index,
            exact_replay=args.exact_replay,
        )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
