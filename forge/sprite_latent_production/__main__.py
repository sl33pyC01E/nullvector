from __future__ import annotations

import argparse
import json
from pathlib import Path

from .checkpoint import load_checkpoint
from .contract import DEFAULT_CORPUS, DEFAULT_OUTPUT, ProductionConfig
from .supervisor import run_calibration, run_supervisor, validate_production_manifest
from .worker import run_segment


def main() -> int:
    parser = argparse.ArgumentParser(description="Segmented production semantic sprite FSQ trainer")
    commands = parser.add_subparsers(dest="command", required=True)
    calibrate = commands.add_parser("calibrate"); calibrate.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS); calibrate.add_argument("--output", type=Path, default=DEFAULT_OUTPUT); calibrate.add_argument("--steps", type=int, default=100)
    train = commands.add_parser("train"); train.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS); train.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    validate = commands.add_parser("validate"); validate.add_argument("checkpoint", type=Path)
    validate_manifest = commands.add_parser("validate-manifest"); validate_manifest.add_argument("manifest", type=Path)
    worker = commands.add_parser("worker"); worker.add_argument("--corpus", type=Path, required=True); worker.add_argument("--output", type=Path, required=True); worker.add_argument("--start-epoch", type=int, required=True); worker.add_argument("--end-epoch", type=int, required=True); worker.add_argument("--resume", type=Path); worker.add_argument("--max-steps", type=int); worker.add_argument("--config-json", required=True)
    args = parser.parse_args()
    if args.command == "calibrate": result = run_calibration(args.output, corpus=args.corpus, steps=args.steps)
    elif args.command == "train": result = run_supervisor(args.output, corpus=args.corpus)
    elif args.command == "validate":
        checkpoint = load_checkpoint(args.checkpoint); result = {"status": "passed", "epoch": checkpoint["epoch"], "global_step": checkpoint["global_step"], "ema_state_sha256": checkpoint["ema_state_sha256"]}
    elif args.command == "validate-manifest":
        manifest = validate_production_manifest(args.manifest); result = {"status": "passed", "manifest_sha256": manifest["manifest_sha256"], "quality_status": manifest["status"], "best_epoch": manifest["best"]["epoch"]}
    else:
        config = ProductionConfig.from_metadata(json.loads(args.config_json)); result = run_segment(corpus_path=args.corpus, output=args.output, config=config, start_epoch=args.start_epoch, end_epoch=args.end_epoch, resume=args.resume, max_steps=args.max_steps)
    print(json.dumps(result, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
