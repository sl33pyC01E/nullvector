from __future__ import annotations
import argparse,json
from pathlib import Path
from .contract import DEFAULT_OUTPUT,TrainingPlan
from .training import train
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT);parser.add_argument("--updates",type=int,default=1000);args=parser.parse_args();print(json.dumps(train(args.output,plan=TrainingPlan(total_updates=args.updates)),indent=2))
if __name__=="__main__":main()
