from __future__ import annotations

import argparse
import json

from . import DEFAULT_OUTPUT, TrainingPlan, evaluate, train


parser=argparse.ArgumentParser();parser.add_argument("command",choices=("train","evaluate"));parser.add_argument("--updates",type=int,default=TrainingPlan().total_updates);args=parser.parse_args();print(json.dumps(train(DEFAULT_OUTPUT,plan=TrainingPlan(total_updates=args.updates)) if args.command=="train" else evaluate(DEFAULT_OUTPUT),indent=2))
