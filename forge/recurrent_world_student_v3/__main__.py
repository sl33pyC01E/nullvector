from __future__ import annotations

import argparse
import json

from .contract import DEFAULT_OUTPUT, TrainingPlan
from .training import train


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=TrainingPlan().total_updates)
    args = parser.parse_args()
    print(json.dumps(train(DEFAULT_OUTPUT, plan=TrainingPlan(total_updates=args.updates)), indent=2))
