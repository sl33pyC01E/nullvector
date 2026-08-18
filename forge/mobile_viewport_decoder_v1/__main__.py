from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contract import DEFAULT_CORPUS, DEFAULT_OUTPUT, DEFAULT_VAE, TrainingConfig
from .training import train

parser = argparse.ArgumentParser()
parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
parser.add_argument("--vae", type=Path, default=DEFAULT_VAE)
parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
parser.add_argument("--steps", type=int, default=4800)
parser.add_argument("--batch-size", type=int, default=12)
arguments = parser.parse_args()
print(json.dumps(train(corpus=arguments.corpus, vae_release=arguments.vae, output=arguments.output, training=TrainingConfig(steps=arguments.steps, batch_size=arguments.batch_size)), indent=2))
