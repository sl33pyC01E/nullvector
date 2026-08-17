from __future__ import annotations

import argparse
import json

from . import DEFAULT_OUTPUT, TrainingPlan, train


parser = argparse.ArgumentParser()
parser.add_argument("--updates", type=int, default=TrainingPlan().total_updates)
parser.add_argument("--segment-updates", type=int, default=TrainingPlan().segment_updates)
parser.add_argument("--batch-size", type=int, default=TrainingPlan().batch_size)
args = parser.parse_args()
print(json.dumps(train(DEFAULT_OUTPUT, plan=TrainingPlan(total_updates=args.updates, segment_updates=args.segment_updates, batch_size=args.batch_size)), indent=2))
