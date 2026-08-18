from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contract import DEFAULT_OUTPUT, TrainingConfig
from .training import train

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
parser.add_argument("--steps", type=int, default=3600)
parser.add_argument("--batch-size", type=int, default=5)
arguments = parser.parse_args()
print(json.dumps(train(arguments.output, training=TrainingConfig(steps=arguments.steps, batch_size=arguments.batch_size)), indent=2))
