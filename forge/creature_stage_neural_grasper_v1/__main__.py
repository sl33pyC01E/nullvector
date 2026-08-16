from __future__ import annotations

import argparse, json
from pathlib import Path
from .contract import TrainingConfig
from .training import train

parser = argparse.ArgumentParser(); parser.add_argument("output", type=Path); parser.add_argument("--steps", type=int, default=1800); parser.add_argument("--batch-size", type=int, default=128); parser.add_argument("--device", default="cuda")
args = parser.parse_args(); print(json.dumps(train(args.output, training=TrainingConfig(steps=args.steps, batch_size=args.batch_size), device=args.device), indent=2))
